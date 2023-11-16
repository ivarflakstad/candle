use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape, WithDType};
use candle_metal_kernels;
use candle_metal_kernels::dispatch::{Queue, QueueAttribute};
use candle_metal_kernels::Kernels;
use half::{bf16, f16};
use metal;
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, HeapDescriptor,
    MTLCommandBufferStatus, MTLResourceOptions, NSUInteger,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, RwLock};

const MAX_BUFFERS: usize = 16;
const MAX_CMD_BUFFERS: usize = MAX_BUFFERS;

#[derive(Debug, Clone)]
pub struct MetalBuffer<T: WithDType> {
    buffer: Buffer,
    size: usize,
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: WithDType> MetalBuffer<T> {
    pub fn new<U: WithDType>(buffer: Buffer, length: usize) -> Self {
        let size = length * T::DTYPE.size_in_bytes();
        Self {
            buffer,
            size,
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetalStorageBuffer {
    U8(MetalBuffer<u8>),
    U32(MetalBuffer<u32>),
    I64(MetalBuffer<i64>),
    BF16(MetalBuffer<bf16>),
    F16(MetalBuffer<f16>),
    F32(MetalBuffer<f32>),
    F64(MetalBuffer<f64>),
}

impl MetalStorageBuffer {
    pub fn read_to_slice<T>(&self, len: usize) -> &[T] {
        let contents_ptr = self.inner().contents() as *const T;
        assert!(!contents_ptr.is_null());
        unsafe { std::slice::from_raw_parts(contents_ptr, len) }
    }

    pub fn read_to_vec<T: Clone>(&self, len: usize) -> Vec<T> {
        self.read_to_slice(len).to_vec()
    }

    pub fn size(&self) -> usize {
        match self {
            MetalStorageBuffer::U8(b) => b.size,
            MetalStorageBuffer::U32(b) => b.size,
            MetalStorageBuffer::I64(b) => b.size,
            MetalStorageBuffer::BF16(b) => b.size,
            MetalStorageBuffer::F16(b) => b.size,
            MetalStorageBuffer::F32(b) => b.size,
            MetalStorageBuffer::F64(b) => b.size,
        }
    }

    pub fn length(&self) -> usize {
        match self {
            MetalStorageBuffer::U8(b) => b.length,
            MetalStorageBuffer::U32(b) => b.length,
            MetalStorageBuffer::I64(b) => b.length,
            MetalStorageBuffer::BF16(b) => b.length,
            MetalStorageBuffer::F16(b) => b.length,
            MetalStorageBuffer::F32(b) => b.length,
            MetalStorageBuffer::F64(b) => b.length,
        }
    }

    pub fn new(buffer: Buffer, length: usize, dtype: DType) -> Self {
        match dtype {
            DType::U8 => MetalStorageBuffer::U8(MetalBuffer::new::<u8>(buffer, length)),
            DType::U32 => MetalStorageBuffer::U32(MetalBuffer::new::<u32>(buffer, length)),
            DType::I64 => MetalStorageBuffer::I64(MetalBuffer::new::<i64>(buffer, length)),
            DType::BF16 => MetalStorageBuffer::BF16(MetalBuffer::new::<bf16>(buffer, length)),
            DType::F16 => MetalStorageBuffer::F16(MetalBuffer::new::<f16>(buffer, length)),
            DType::F32 => MetalStorageBuffer::F32(MetalBuffer::new::<f32>(buffer, length)),
            DType::F64 => MetalStorageBuffer::F64(MetalBuffer::new::<f64>(buffer, length)),
        }
    }

    pub fn inner(&self) -> Buffer {
        match self {
            MetalStorageBuffer::U8(metal_buffer) => metal_buffer.buffer.clone(),
            MetalStorageBuffer::U32(metal_buffer) => metal_buffer.buffer.clone(),
            MetalStorageBuffer::I64(metal_buffer) => metal_buffer.buffer.clone(),
            MetalStorageBuffer::BF16(metal_buffer) => metal_buffer.buffer.clone(),
            MetalStorageBuffer::F16(metal_buffer) => metal_buffer.buffer.clone(),
            MetalStorageBuffer::F32(metal_buffer) => metal_buffer.buffer.clone(),
            MetalStorageBuffer::F64(metal_buffer) => metal_buffer.buffer.clone(),
        }
    }
}

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_metal_kernels::MetalKernelError),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    device: Arc<metal::Device>,
    command_queue: Arc<metal::CommandQueue>,
    heap: Arc<RwLock<metal::Heap>>,
    fence: Arc<metal::Fence>,
    commands: Arc<Mutex<VecDeque<Command>>>,
    buffers: Arc<Mutex<HashMap<usize, MetalStorageBuffer>>>,
    kernels: Arc<candle_metal_kernels::Kernels>,
    queue: Arc<Queue>,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.device.registry_id())
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = metal::DeviceRef;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

#[derive(Debug, Clone)]
pub struct Command {
    buffer: Arc<CommandBuffer>,
    encoder: Arc<ComputeCommandEncoder>,
}

impl Command {
    fn new(cb: CommandBuffer, ce: ComputeCommandEncoder) -> Command {
        Self {
            buffer: Arc::new(cb),
            encoder: Arc::new(ce),
        }
    }
}

impl MetalDevice {
    pub fn id(&self) -> NSUInteger {
        self.registry_id()
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    pub fn new_command(&self) -> Command {
        let mut commands = self.commands.lock().unwrap();

        let mut keep = VecDeque::new();
        for cmd in commands.iter() {
            if (cmd.buffer.status() as u32) == (MTLCommandBufferStatus::Enqueued as u32) {
                keep.push_back(cmd.clone());
            }
        }

        let to_create = MAX_CMD_BUFFERS - keep.len();
        for _ in 0..to_create {
            let cmd_buffer = self.command_queue.new_command_buffer().to_owned();
            cmd_buffer.enqueue();
            let cmp_encoder = cmd_buffer.new_compute_command_encoder().to_owned();
            keep.push_back(Command::new(cmd_buffer, cmp_encoder));
        }

        *commands = keep;

        commands.pop_front().unwrap()
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let cmd_buffers = self
            .commands
            .lock()
            .map_err(|_| MetalError::Message("Could not lock command buffer queue".to_string()))?;
        cmd_buffers.iter().for_each(|cmd| {
            let buf = cmd.buffer.clone();
            self.queue.exec_async(move || {
                match buf.status() {
                    MTLCommandBufferStatus::NotEnqueued => {
                        // cmdb.buffer.enqueue();
                        // cmdb.encoder.end_encoding();
                        // cmdb.buffer.commit();
                        // cmdb.buffer.wait_until_completed();
                    }
                    MTLCommandBufferStatus::Enqueued => {
                        // cmdb.encoder.end_encoding();
                        // cmdb.buffer.commit();
                        // cmdb.buffer.wait_until_completed();
                    }
                    MTLCommandBufferStatus::Committed => buf.wait_until_completed(),
                    MTLCommandBufferStatus::Scheduled => buf.wait_until_completed(),
                    MTLCommandBufferStatus::Completed => {}
                    MTLCommandBufferStatus::Error => todo!("Command buffer error"),
                }
            });
        });
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    pub fn get_buffer(&self, element_count: usize, dtype: DType) -> Result<Buffer> {
        let size = element_count * dtype.size_in_bytes();
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(b) = buffers.get(&size) {
            return Ok(b.inner());
        }

        let mut heap = self.heap.write().unwrap();
        let used_size = heap.used_size();
        let ns_size = size as NSUInteger;
        if heap.max_available_size_with_alignment(0) < used_size + ns_size {
            let desc = HeapDescriptor::new();
            desc.set_size(used_size + ns_size);
            *heap = self.device.new_heap(&desc);
        }
        let b = heap
            .new_buffer(ns_size, MTLResourceOptions::StorageModeShared)
            .unwrap();

        buffers.insert(
            size,
            MetalStorageBuffer::new(b.clone(), element_count, dtype),
        );

        return Ok(b);
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T], dtype: DType) -> Result<MetalStorageBuffer> {
        let b = self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            (data.len() * dtype.size_in_bytes()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalStorageBuffer::new(b, data.len(), dtype))
    }
}

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: MetalStorageBuffer,
    device: MetalDevice,
    dtype: DType,
}

impl MetalStorage {
    pub fn read_to_slice<T>(&self, len: usize) -> &[T] {
        let contents_ptr = self.buffer.inner().contents() as *const T;
        assert!(!contents_ptr.is_null());
        unsafe { std::slice::from_raw_parts(contents_ptr, len) }
    }

    pub fn read_to_vec<T: Clone>(&self, len: usize) -> Vec<T> {
        self.read_to_slice(len).to_vec()
    }
}

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.device.wait_until_completed()?;
        let length = self.buffer.inner().length() as usize;
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.read_to_vec(length))),
            DType::U32 => Ok(CpuStorage::U32(self.read_to_vec(length / 4))),
            DType::I64 => Ok(CpuStorage::I64(self.read_to_vec(length / 8))),
            DType::F16 => Ok(CpuStorage::F16(self.read_to_vec(length / 2))),
            DType::BF16 => Ok(CpuStorage::BF16(self.read_to_vec(length / 2))),
            DType::F32 => Ok(CpuStorage::F32(self.read_to_vec(length / 4))),
            DType::F64 => Ok(CpuStorage::F64(self.read_to_vec(length / 8))),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();

        let el = layout.shape().elem_count();
        let dtype = self.dtype;

        let mut output_buffer = device.get_buffer(el, self.dtype)?;
        let buffer = MetalStorageBuffer::new(output_buffer.clone(), el, self.dtype);
        let input_buffer = self.buffer.inner();

        let mtl_device = device.device.clone();

        let layout = layout.clone();

        self.device.queue.exec_async(move || {
            let command = device.new_command();
            if layout.is_contiguous() && layout.start_offset() == 0 {
                let name = match dtype {
                    DType::F32 => "affine_float",
                    DType::F16 => "affine_half",
                    dtype => todo!("Affine {dtype:?}"),
                };
                candle_metal_kernels::call_affine(
                    &mtl_device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    name,
                    el,
                    &input_buffer,
                    &mut output_buffer,
                    mul as f32,
                    add as f32,
                )
                .unwrap();
            } else {
                let name = match dtype {
                    DType::F32 => "affine_float_strided",
                    DType::F16 => "affine_half_strided",
                    dtype => todo!("Affine {dtype:?}"),
                };
                candle_metal_kernels::call_affine_strided(
                    &device.device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    name,
                    layout.dims(),
                    &input_buffer,
                    layout.stride(),
                    layout.start_offset() * dtype.size_in_bytes(),
                    &mut output_buffer,
                    mul as f32,
                    add as f32,
                )
                .unwrap();
            }
            command.buffer.commit();
        });
        Ok(Self {
            buffer,
            device: self.device().clone(),
            dtype,
        })
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        assert!(sum_dims.len() == 1);
        assert!(sum_dims[0] == layout.shape().rank() - 1);
        assert!(layout.is_contiguous());
        assert!(layout.start_offset() == 0);
        let device = self.device.clone();
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !sum_dims.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in sum_dims.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }

        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let (name, check_empty, return_index) = match (op, self.dtype) {
            (ReduceOp::Sum, DType::F32) => ("fast_sum_float", false, false),
            (ReduceOp::Min, DType::F32) => ("fast_min_float", true, false),
            (ReduceOp::Max, DType::F32) => ("fast_max_float", true, false),
            (ReduceOp::ArgMin, DType::F32) => ("fast_argmin_float", true, true),
            (ReduceOp::ArgMax, DType::F32) => ("fast_argmax_float", true, true),
            _ => todo!("Reduce op for non float"),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dtype = if return_index { DType::U32 } else { self.dtype };
        let input_buffer = self.buffer.inner();
        let mut output_buffer = device.get_buffer(dst_el, dtype)?;
        let buffer = MetalStorageBuffer::new(output_buffer.clone(), dst_el, self.dtype);
        self.device.queue.exec_async(move || {
            let command = device.new_command();
            candle_metal_kernels::call_reduce_contiguous(
                &device.device,
                &command.encoder,
                &device.heap.read().unwrap(),
                &device.fence,
                &device.kernels,
                name,
                src_el,
                dst_el,
                &input_buffer,
                &mut output_buffer,
            )
            .map_err(MetalError::from)
            .unwrap();
            command.buffer.commit();
        });
        Ok(Self {
            buffer,
            device: self.device().clone(),
            dtype,
        })
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let device = self.device().clone();
        let layout = layout.clone();
        let shape = layout.shape();
        let el_count = shape.elem_count();

        let original_dtype = self.dtype;
        let input_buffer = self.buffer.inner();
        let mut output_buffer = device.get_buffer(el_count, dtype)?;
        let buffer = MetalStorageBuffer::new(output_buffer.clone(), el_count, self.dtype);

        self.device.queue.exec_async(move || {
            let command = device.new_command();
            if layout.is_contiguous() {
                let kernel_name = match (original_dtype, dtype) {
                    (DType::U32, DType::F32) => "cast_u32_f32",
                    (DType::F32, DType::F16) => "cast_f32_f16",
                    (DType::F16, DType::F32) => "cast_f16_f32",
                    (left, right) => todo!("to dtype {left:?} - {right:?}"),
                };
                candle_metal_kernels::call_cast_contiguous(
                    &device.device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    kernel_name,
                    el_count,
                    &input_buffer,
                    &mut output_buffer,
                )
                .map_err(MetalError::from)
                .unwrap();
            } else {
                let kernel_name = match (original_dtype, dtype) {
                    (DType::U32, DType::F32) => "cast_u32_f32_strided",
                    (DType::F32, DType::F16) => "cast_f32_f16_strided",
                    (DType::F16, DType::F32) => "cast_f16_f32_strided",
                    (left, right) => todo!("to dtype {left:?} - {right:?}"),
                };
                candle_metal_kernels::call_cast_strided(
                    &device.device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    kernel_name,
                    layout.dims(),
                    &input_buffer,
                    layout.stride(),
                    layout.start_offset() * original_dtype.size_in_bytes(),
                    &mut output_buffer,
                )
                .map_err(MetalError::from)
                .unwrap();
            }

            command.buffer.commit();
        });

        Ok(Self {
            buffer,
            device: self.device().clone(),
            dtype,
        })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let dtype = self.dtype;
        let layout = layout.clone();
        let shape = layout.shape();
        let el_count = shape.elem_count();

        let input_buffer = self.buffer.inner();
        let mut output_buffer = device.get_buffer(el_count, dtype)?;
        let buffer = MetalStorageBuffer::new(output_buffer.clone(), el_count, self.dtype);

        self.device.queue.exec_async(move || {
            let command = device.new_command();
            if layout.is_contiguous() && layout.start_offset() == 0 {
                use candle_metal_kernels::unary::contiguous;

                let kernel_name = match (B::KERNEL, dtype) {
                    ("ucos", DType::F32) => contiguous::cos::FLOAT,
                    ("usin", DType::F32) => contiguous::sin::FLOAT,
                    ("usqr", DType::F32) => contiguous::sqr::FLOAT,
                    ("usqrt", DType::F32) => contiguous::sqrt::FLOAT,
                    ("uneg", DType::F32) => contiguous::neg::FLOAT,
                    ("uexp", DType::F32) => contiguous::exp::FLOAT,
                    ("ulog", DType::F32) => contiguous::log::FLOAT,
                    ("ugelu", DType::F32) => contiguous::gelu::FLOAT,
                    ("ugelu_erf", DType::F32) => contiguous::gelu_erf::FLOAT,
                    ("uerf", DType::F32) => contiguous::erf::FLOAT,
                    ("uceil", DType::F32) => contiguous::ceil::FLOAT,
                    ("ufloor", DType::F32) => contiguous::floor::FLOAT,
                    ("uround", DType::F32) => contiguous::round::FLOAT,
                    ("ucos", DType::F16) => contiguous::cos::HALF,
                    ("usin", DType::F16) => contiguous::sin::HALF,
                    ("usqr", DType::F16) => contiguous::sqr::HALF,
                    ("usqrt", DType::F16) => contiguous::sqrt::HALF,
                    ("uneg", DType::F16) => contiguous::neg::HALF,
                    ("uexp", DType::F16) => contiguous::exp::HALF,
                    ("ulog", DType::F16) => contiguous::log::HALF,
                    ("ugelu", DType::F16) => contiguous::gelu::HALF,
                    ("ugelu_erf", DType::F16) => contiguous::gelu_erf::HALF,
                    ("uerf", DType::F16) => contiguous::erf::HALF,
                    ("uceil", DType::F16) => contiguous::ceil::HALF,
                    ("ufloor", DType::F16) => contiguous::floor::HALF,
                    ("uround", DType::F16) => contiguous::round::HALF,
                    (name, dtype) => todo!("Match {name} - {dtype:?}"),
                };
                candle_metal_kernels::call_unary_contiguous(
                    &device.device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    kernel_name,
                    el_count,
                    &input_buffer,
                    &mut output_buffer,
                )
                .map_err(MetalError::from)
                .unwrap();
            } else {
                use candle_metal_kernels::unary::strided;
                let kernel_name = match (B::KERNEL, dtype) {
                    ("ucos", DType::F32) => strided::cos::FLOAT,
                    ("usin", DType::F32) => strided::sin::FLOAT,
                    ("usqr", DType::F32) => strided::sqr::FLOAT,
                    ("usqrt", DType::F32) => strided::sqrt::FLOAT,
                    ("uneg", DType::F32) => strided::neg::FLOAT,
                    ("uexp", DType::F32) => strided::exp::FLOAT,
                    ("ulog", DType::F32) => strided::log::FLOAT,
                    ("ugelu", DType::F32) => strided::gelu::FLOAT,
                    ("ugelu_erf", DType::F32) => strided::gelu_erf::FLOAT,
                    ("uerf", DType::F32) => strided::erf::FLOAT,
                    ("uceil", DType::F32) => strided::ceil::FLOAT,
                    ("ufloor", DType::F32) => strided::floor::FLOAT,
                    ("uround", DType::F32) => strided::round::FLOAT,
                    ("ucos", DType::F16) => strided::cos::HALF,
                    ("usin", DType::F16) => strided::sin::HALF,
                    ("usqr", DType::F16) => strided::sqr::HALF,
                    ("usqrt", DType::F16) => strided::sqrt::HALF,
                    ("uneg", DType::F16) => strided::neg::HALF,
                    ("uexp", DType::F16) => strided::exp::HALF,
                    ("ulog", DType::F16) => strided::log::HALF,
                    ("ugelu", DType::F16) => strided::gelu::HALF,
                    ("ugelu_erf", DType::F16) => strided::gelu_erf::HALF,
                    ("uerf", DType::F16) => strided::erf::HALF,
                    ("uceil", DType::F16) => strided::ceil::HALF,
                    ("ufloor", DType::F16) => strided::floor::HALF,
                    ("uround", DType::F16) => strided::round::HALF,
                    (name, dtype) => todo!("Match {name} - {dtype:?}"),
                };
                candle_metal_kernels::call_unary_strided(
                    &device.device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    kernel_name,
                    layout.dims(),
                    &input_buffer,
                    layout.stride(),
                    layout.start_offset() * dtype.size_in_bytes(),
                    &mut output_buffer,
                    0,
                )
                .map_err(MetalError::from)
                .unwrap();
            }

            command.buffer.commit();
        });

        Ok(Self {
            buffer,
            device: self.device().clone(),
            dtype,
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let lhs_dtype = self.dtype;
        let rhs_dtype = rhs.dtype;
        let lhs_l = lhs_l.clone();
        let rhs_l = rhs_l.clone();
        let shape = lhs_l.shape();
        let el_count = shape.elem_count();
        let lhs_buffer = self.buffer.inner();
        let rhs_buffer = rhs.buffer.inner();
        let mut output_buffer = device.get_buffer(el_count, lhs_dtype)?;
        let buffer = MetalStorageBuffer::new(output_buffer.clone(), el_count, self.dtype);

        self.device.queue.exec_async(move || {
            let command = device.new_command();
            if (lhs_l.is_contiguous() && lhs_l.start_offset() == 0)
                && (rhs_l.is_contiguous() && rhs_l.start_offset() == 0)
            {
                use candle_metal_kernels::binary::contiguous;

                let kernel_name = match (B::KERNEL, lhs_dtype) {
                    ("add", DType::F32) => contiguous::add::FLOAT,
                    ("badd", DType::F32) => contiguous::add::FLOAT,
                    ("sub", DType::F32) => contiguous::sub::FLOAT,
                    ("bsub", DType::F32) => contiguous::sub::FLOAT,
                    ("mul", DType::F32) => contiguous::mul::FLOAT,
                    ("bmul", DType::F32) => contiguous::mul::FLOAT,
                    ("div", DType::F32) => contiguous::div::FLOAT,
                    ("bdiv", DType::F32) => contiguous::div::FLOAT,
                    ("add", DType::F16) => contiguous::add::HALF,
                    ("badd", DType::F16) => contiguous::add::HALF,
                    ("sub", DType::F16) => contiguous::sub::HALF,
                    ("bsub", DType::F16) => contiguous::sub::HALF,
                    ("mul", DType::F16) => contiguous::mul::HALF,
                    ("bmul", DType::F16) => contiguous::mul::HALF,
                    ("div", DType::F16) => contiguous::div::HALF,
                    ("bdiv", DType::F16) => contiguous::div::HALF,
                    (name, dtype) => todo!("Match {name} - {dtype:?}"),
                };
                candle_metal_kernels::call_binary_contiguous(
                    &device.device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    kernel_name,
                    el_count,
                    &lhs_buffer,
                    &rhs_buffer,
                    &mut output_buffer,
                )
                .map_err(MetalError::from)
                .unwrap();
            } else {
                use candle_metal_kernels::binary::strided;

                let kernel_name = match (B::KERNEL, lhs_dtype) {
                    ("badd", DType::F32) => strided::add::FLOAT,
                    ("bsub", DType::F32) => strided::sub::FLOAT,
                    ("bmul", DType::F32) => strided::mul::FLOAT,
                    ("bdiv", DType::F32) => strided::div::FLOAT,
                    ("badd", DType::F16) => strided::add::HALF,
                    ("bsub", DType::F16) => strided::sub::HALF,
                    ("bmul", DType::F16) => strided::mul::HALF,
                    ("bdiv", DType::F16) => strided::div::HALF,
                    (name, dtype) => todo!("Match {name} - {dtype:?}"),
                };
                candle_metal_kernels::call_binary_strided(
                    &device.device,
                    &command.encoder,
                    &device.heap.read().unwrap(),
                    &device.fence,
                    &device.kernels,
                    kernel_name,
                    lhs_l.dims(),
                    &lhs_buffer,
                    lhs_l.stride(),
                    lhs_l.start_offset() * lhs_dtype.size_in_bytes(),
                    &rhs_buffer,
                    rhs_l.stride(),
                    rhs_l.start_offset() * rhs_dtype.size_in_bytes(),
                    &mut output_buffer,
                )
                .map_err(MetalError::from)
                .unwrap();
            }
            command.buffer.commit();
        });
        Ok(Self {
            buffer,
            device: self.device().clone(),
            dtype: lhs_dtype,
        })
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device.clone();
        let layout = layout.clone();
        let t_l = t_l.clone();
        let f_l = f_l.clone();
        let shape = t_l.shape().clone();
        let el = shape.elem_count();
        let self_dtype = self.dtype;
        let dtype = t.dtype;
        let f_dtype = f.dtype;
        let input_buffer = self.buffer.inner();
        let t_buffer = t.buffer().inner();
        let f_buffer = f.buffer().inner();

        let mut output_buffer = self.device.get_buffer(el, dtype)?;
        let buffer = MetalStorageBuffer::new(output_buffer.clone(), el, self.dtype);

        self.device.queue.exec_async(move || {
            let command = device.new_command();
            candle_metal_kernels::call_where_cond_strided(
                &device.device,
                &command.encoder,
                &device.heap.read().unwrap(),
                &device.fence,
                &device.kernels,
                "where_u8_f32",
                shape.dims(),
                &input_buffer,
                (
                    layout.stride(),
                    layout.start_offset() * self_dtype.size_in_bytes(),
                ),
                &t_buffer,
                (&t_l.stride(), t_l.start_offset() * dtype.size_in_bytes()),
                &f_buffer,
                (&f_l.stride(), f_l.start_offset() * f_dtype.size_in_bytes()),
                &mut output_buffer,
            )
            .map_err(MetalError::from)
            .unwrap();
            command.buffer.commit();
        });
        Ok(Self {
            buffer,
            device: self.device.clone(),
            dtype,
        })
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv2D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        todo!()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        todo!()
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        assert!(src_l.is_contiguous());
        assert!(src_l.start_offset() == 0);
        assert!(ids_l.is_contiguous());
        assert!(ids_l.start_offset() == 0);
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        let dst_el = ids_el * left_size * right_size;
        let dtype = self.dtype;
        let device = self.device().clone();
        let name = match (ids.dtype, self.dtype) {
            (DType::U32, DType::F32) => "is_u32_f32",
            (DType::U32, DType::F16) => "is_u32_f16",
            (left, right) => todo!("index select metal {left:?} {right:?}"),
        };
        let kernels = self.device.kernels.clone();
        let input_buffer = self.buffer.inner();
        let ids_buffer = ids.buffer.inner();
        let mut output_buffer = device.get_buffer(dst_el, dtype)?;
        let buffer = MetalStorageBuffer::new(output_buffer.clone(), dst_el, dtype);

        let src_l = src_l.clone();

        self.device.queue.exec_async(move || {
            let command = device.new_command();
            candle_metal_kernels::call_index_select(
                &device.device,
                &command.encoder,
                &device.heap.read().unwrap(),
                &device.fence,
                &kernels,
                name,
                src_l.dims(),
                ids_el,
                dim,
                &input_buffer,
                &ids_buffer,
                &mut output_buffer,
            )
            .map_err(MetalError::from)
            .unwrap();
            command.buffer.commit();
        });

        Ok(Self {
            buffer,
            device: self.device().clone(),
            dtype,
        })
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let out_buffer = self.device.get_buffer(elem_count, self.dtype)?;
        let buffer = MetalStorageBuffer::new(out_buffer.clone(), elem_count, self.dtype);

        // Create descriptors
        use metal::mps::matrix::*;

        let (type_id, size) = match self.dtype {
            DType::F32 => (
                metal::mps::MPS_FLOATBIT_ENCODING | 32,
                core::mem::size_of::<f32>() as NSUInteger,
            ),
            DType::F16 => (
                metal::mps::MPS_FLOATBIT_ENCODING | 16,
                core::mem::size_of::<f16>() as NSUInteger,
            ),
            dtype => todo!("Dtype for matmul {dtype:?} is not supported"),
        };

        let lhs_stride = lhs_l.stride().to_vec();
        let rhs_stride = rhs_l.stride().to_vec();
        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

        let lhs_buffer = self.buffer.inner();
        let lhs_l_start_offset = lhs_l.start_offset();
        let lhs_l_dims = lhs_l.dims().to_vec();

        let rhs_buffer = rhs.buffer.inner();
        let rhs_l_start_offset = rhs_l.start_offset();
        let rhs_l_dims = lhs_l.dims().to_vec();

        let device = self.device.clone();

        self.device.queue.exec_async(move || {
            let command = device.new_command();
            command.encoder.end_encoding();
            // The a tensor has dims batching, k, n (rhs)
            let transpose_left = if lhs_m1 == 1 && lhs_m2 == k {
                false
            } else if lhs_m1 == m && lhs_m2 == 1 {
                true
            } else {
                Err(MetalError::MatMulNonContiguous {
                    lhs_stride: lhs_stride.clone(),
                    rhs_stride: rhs_stride.clone(),
                    mnk: (m, n, k),
                })
                .unwrap()
            };
            let transpose_right = if rhs_m1 == 1 && rhs_m2 == n {
                false
            } else if rhs_m1 == k && rhs_m2 == 1 {
                true
            } else {
                Err(MetalError::MatMulNonContiguous {
                    lhs_stride: lhs_stride.clone(),
                    rhs_stride: rhs_stride.clone(),
                    mnk: (m, n, k),
                })
                .unwrap()
            };
            let stride_left: u64 = match lhs_stride[..lhs_stride.len() - 2] {
                [s1, stride] if s1 == stride * lhs_l_dims[1] => stride,
                [stride] => stride,
                [] => m * k,
                _ => Err(MetalError::MatMulNonContiguous {
                    lhs_stride: lhs_stride.clone(),
                    rhs_stride: rhs_stride.clone(),
                    mnk: (m, n, k),
                })
                .unwrap(),
            } as u64;
            let stride_right: u64 = match rhs_stride[..rhs_stride.len() - 2] {
                [s1, stride] if s1 == stride * rhs_l_dims[1] => stride,
                [stride] => stride,
                [] => n * k,
                _ => Err(MetalError::MatMulNonContiguous {
                    lhs_stride: lhs_stride.clone(),
                    rhs_stride: rhs_stride.clone(),
                    mnk: (m, n, k),
                })
                .unwrap(),
            } as u64;

            let b = b as NSUInteger;
            let m = m as NSUInteger;
            let n = n as NSUInteger;
            let k = k as NSUInteger;

            let left_descriptor = if transpose_left {
                MatrixDescriptor::init_single(k, m, m * size, type_id)
            } else {
                MatrixDescriptor::init_single(m, k, k * size, type_id)
            };
            let right_descriptor = if transpose_right {
                MatrixDescriptor::init_single(n, k, k * size, type_id)
            } else {
                MatrixDescriptor::init_single(k, n, n * size, type_id)
            };
            let result_descriptor = MatrixDescriptor::init_single(m, n, n * size, type_id);

            for bi in 0..b {
                // Create matrix objects
                let left_matrix = Matrix::init_with_buffer_descriptor(
                    &lhs_buffer,
                    (bi * stride_left + lhs_l_start_offset as u64) * size,
                    &left_descriptor,
                )
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })
                .unwrap();
                let right_matrix = Matrix::init_with_buffer_descriptor(
                    &rhs_buffer,
                    (bi * stride_right + rhs_l_start_offset as u64) * size,
                    &right_descriptor,
                )
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })
                .unwrap();

                let result_matrix = Matrix::init_with_buffer_descriptor(
                    &out_buffer,
                    bi * m * n * size,
                    &result_descriptor,
                )
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })
                .unwrap();

                let alpha = 1.0f64;
                let beta = 0.0f64;
                // Create kernel
                let matrix_multiplication = MatrixMultiplication::init(
                    &device,
                    transpose_left,
                    transpose_right,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                )
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })
                .unwrap();
                // Encode kernel to command buffer
                matrix_multiplication.encode_to_command_buffer(
                    &command.buffer,
                    &left_matrix,
                    &right_matrix,
                    &result_matrix,
                );
            }

            command.buffer.commit();
        });

        Ok(Self {
            buffer,
            device: self.device.clone(),
            dtype: self.dtype(),
        })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        let kernel_name = match self.dtype {
            DType::F32 => candle_metal_kernels::unary::strided::copy::FLOAT,
            DType::F16 => candle_metal_kernels::unary::strided::copy::HALF,
            DType::BF16 => candle_metal_kernels::unary::strided::copy::BFLOAT,
            DType::U32 => candle_metal_kernels::unary::strided::copy::U32,
            dtype => todo!("copy_strided not implemented for {dtype:?}"),
        };

        let metal_device = self.device().clone();
        let buffer = self.buffer.inner();

        let src_l = src_l.clone();
        let dtype = self.dtype.clone();

        let dst_dtype = dst.dtype().clone();
        let mut dst_buffer = dst.buffer.inner();

        self.device.queue.exec_async(move || {
            let command = metal_device.new_command();
            candle_metal_kernels::call_unary_strided(
                &metal_device.device,
                &command.encoder,
                &metal_device.heap.read().unwrap(),
                &metal_device.fence,
                &metal_device.kernels,
                kernel_name,
                src_l.dims(),
                &buffer,
                src_l.stride(),
                src_l.start_offset() * dtype.size_in_bytes(),
                &mut dst_buffer,
                dst_offset * dst_dtype.size_in_bytes(),
            )
            .map_err(MetalError::from)
            .unwrap();
            command.buffer.commit();
        });

        Ok(())
    }
}

impl MetalStorage {
    pub fn new(buffer: MetalStorageBuffer, device: MetalDevice, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            dtype,
        }
    }

    pub fn buffer(&self) -> &MetalStorageBuffer {
        &self.buffer
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = metal::Device::all().swap_remove(ordinal);

        let command_queue = device.new_command_queue();

        let descriptor = HeapDescriptor::new();
        let mut size =
            device.heap_buffer_size_and_align(100_000_000, MTLResourceOptions::StorageModeShared);
        size.size += (size.size & (size.align - 1)) + size.align;
        descriptor.set_size(size.size);
        descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        let heap = Arc::new(RwLock::new(device.new_heap(&descriptor)));
        let fence = Arc::new(device.new_fence());

        let commands = (0..MAX_CMD_BUFFERS)
            .map(|_| {
                let cmd_buffer = command_queue.new_owned_command_buffer();
                cmd_buffer.enqueue();

                let cmp_encoder = cmd_buffer.new_compute_command_encoder().to_owned();

                Command::new(cmd_buffer, cmp_encoder)
            })
            .collect();
        let kernels = Arc::new(Kernels::new());
        Ok(Self {
            device: Arc::new(device),
            heap,
            fence,
            command_queue: Arc::new(command_queue),
            commands: Arc::new(Mutex::new(commands)),
            buffers: Default::default(),
            kernels,
            queue: Arc::new(Queue::create("com.candle.metal", QueueAttribute::Serial)),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal {
            gpu_id: self.registry_id() as usize,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.device.registry_id() == rhs.device.registry_id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let buffer = self.get_buffer(shape.elem_count(), dtype)?;
        Ok(MetalStorage {
            buffer: MetalStorageBuffer::new(buffer, shape.elem_count(), dtype),
            device: self.clone(),
            dtype,
        })
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // TODO Is there a faster way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.ones_impl(shape, dtype)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let buffer = match storage {
            CpuStorage::U8(storage) => self.new_buffer_with_data(storage, DType::U8)?,
            CpuStorage::U32(storage) => self.new_buffer_with_data(storage, DType::U32)?,
            CpuStorage::I64(storage) => self.new_buffer_with_data(storage, DType::I64)?,
            CpuStorage::BF16(storage) => self.new_buffer_with_data(storage, DType::BF16)?,
            CpuStorage::F16(storage) => self.new_buffer_with_data(storage, DType::F16)?,
            CpuStorage::F32(storage) => self.new_buffer_with_data(storage, DType::F32)?,
            // TODO: Metal does not support double.
            CpuStorage::F64(storage) => self.new_buffer_with_data(storage, DType::F64)?,
        };
        Ok(Self::Storage {
            buffer,
            device: self.clone(),
            dtype: storage.dtype(),
        })
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        // TODO is there a better way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, mean, stddev)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        // TODO is there a better way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, stddev)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        todo!("set_seed")
    }
}
