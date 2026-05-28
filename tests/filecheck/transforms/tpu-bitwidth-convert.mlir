// RUN: xdsl-opt --split-input-file -p tpu-bitwidth-convert %s | filecheck %s

// bf16 reduction IS converted to f32 compute
%s = "test.op"() : () -> vector<128x128xbf16>
%a = "test.op"() : () -> vector<128xbf16>
%r = vector.multi_reduction <add>, %s, %a [1] : vector<128x128xbf16> to vector<128xbf16>
"test.op"(%r) : (vector<128xbf16>) -> ()
// CHECK:      %s = "test.op"() : () -> vector<128x128xbf16>
// CHECK-NEXT: %a = "test.op"() : () -> vector<128xbf16>
// CHECK-NEXT: %{{.*}} = arith.extf %s : vector<128x128xbf16> to vector<128x128xf32>
// CHECK-NEXT: %{{.*}} = arith.extf %a : vector<128xbf16> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [1] : vector<128x128xf32> to vector<128xf32>
// CHECK-NEXT: %{{.*}} = arith.truncf %{{.*}} : vector<128xf32> to vector<128xbf16>
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<128xbf16>) -> ()

// -----

// f32 reduction is NOT converted (no bf16, nothing to do)
%fs = "test.op"() : () -> vector<128x128xf32>
%fa = "test.op"() : () -> vector<128xf32>
%fr = vector.multi_reduction <add>, %fs, %fa [1] : vector<128x128xf32> to vector<128xf32>
"test.op"(%fr) : (vector<128xf32>) -> ()
// CHECK:      %fs = "test.op"() : () -> vector<128x128xf32>
// CHECK-NEXT: %fa = "test.op"() : () -> vector<128xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %fs, %fa [1] : vector<128x128xf32> to vector<128xf32>
// CHECK-NEXT: "test.op"(%{{.*}}) : (vector<128xf32>) -> ()
