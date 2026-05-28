// RUN: xdsl-opt --split-input-file -p tpu-canonicalize %s | filecheck %s

// BitcastVregOp: no-op (X -> X) is removed (fold).
%v0 = "test.op"() : () -> vector<4x8xf32>
%bc_noop = tpu.bitcast_vreg %v0 : vector<4x8xf32> -> vector<4x8xf32>
"test.op"(%bc_noop) : (vector<4x8xf32>) -> ()
// CHECK:      %v0 = "test.op"() : () -> vector<4x8xf32>
// CHECK-NEXT: "test.op"(%v0) : (vector<4x8xf32>) -> ()

// -----

// ReshapeOp: no-op reshape (same type) is removed (fold).
%v2 = "test.op"() : () -> vector<4x8xf32>
%rs_noop = tpu.reshape %v2 : vector<4x8xf32> -> vector<4x8xf32>
"test.op"(%rs_noop) : (vector<4x8xf32>) -> ()
// CHECK:      %v2 = "test.op"() : () -> vector<4x8xf32>
// CHECK-NEXT: "test.op"(%v2) : (vector<4x8xf32>) -> ()

// -----

// MemRefBitcastOp: no-op (X -> X) is removed (fold).
%mb = "test.op"() : () -> memref<4x8xf32>
%mbc = tpu.memref_bitcast %mb : memref<4x8xf32> -> memref<4x8xf32>
"test.op"(%mbc) : (memref<4x8xf32>) -> ()
// CHECK:      %mb = "test.op"() : () -> memref<4x8xf32>
// CHECK-NEXT: "test.op"(%mb) : (memref<4x8xf32>) -> ()

// -----

// MemRefSliceOp: no-op (unchanged type, all base indices constant 0) is removed (fold).
%ms = "test.op"() : () -> memref<4x8xf32>
%zero = arith.constant 0 : i32
%msl = tpu.memref_slice %ms[%zero, %zero] : memref<4x8xf32> -> memref<4x8xf32>
"test.op"(%msl) : (memref<4x8xf32>) -> ()
// CHECK:      %ms = "test.op"() : () -> memref<4x8xf32>
// CHECK-NEXT: "test.op"(%ms) : (memref<4x8xf32>) -> ()

// -----

// EraseLayoutOp: no-op (operand type == result type) is removed (fold).
%me = "test.op"() : () -> memref<4x8xf32>
%mer = tpu.erase_memref_layout %me : memref<4x8xf32> -> memref<4x8xf32>
"test.op"(%mer) : (memref<4x8xf32>) -> ()
// CHECK:      %me = "test.op"() : () -> memref<4x8xf32>
// CHECK-NEXT: "test.op"(%me) : (memref<4x8xf32>) -> ()
