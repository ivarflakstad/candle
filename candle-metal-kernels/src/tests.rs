use crate::*;
use metal::{Device, DeviceRef, MTLDataType, MTLDispatchType, MTLResourceOptions};
use std::{mem, slice};

#[test]
fn test_mul() {
    let device: &DeviceRef = &Device::system_default().expect("No device found");
    let lib = device.new_library_with_data(MUL).unwrap();
    let function = lib.get_function("mul", None).unwrap();
    println!("dot_product: {:?}", function);

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();
    let v = &[[1., 2., 3., 4.], [5., 6., 7., 8.]];
    let w = &[[2., 3., 4., 5.], [6., 7., 8., 9.]];
    let length = v.len() as u64;
    let size = length * mem::size_of::<u32>() as u64;
    assert_eq!(v.len(), w.len());

    let buffer_a = device.new_buffer_with_data(
        unsafe { mem::transmute(v.as_ptr()) },
        size,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = device.new_buffer_with_data(
        unsafe { mem::transmute(w.as_ptr()) },
        size,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_result = device.new_buffer(size, MTLResourceOptions::StorageModeShared);

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder =
        command_buffer.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffers(
        0,
        &[Some(&buffer_a), Some(&buffer_b), Some(&buffer_result)],
        &[0; 3],
    );

    let grid_size = metal::MTLSize::new(length, 1, 1);
    let threadgroup_size = metal::MTLSize::new(length, 4, 1);
    compute_encoder.dispatch_threads(grid_size, threadgroup_size);
    compute_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let ptr = buffer_result.contents() as *const u32;
    println!("ptr: {:?}", ptr);
    let len = buffer_result.length() as usize / mem::size_of::<&[f64]>();
    println!("len: {:?}", len);
    let slice = unsafe { slice::from_raw_parts(ptr, len) };
    println!("slice: {:?}", slice);
    let result = slice.to_vec();
    println!("Results!");
    println!("{:?}", result);
}
