module neural_net_top (
    input clk_100MHz,
    input rst_n,
    input [7:0] audio_input [0:124][0:129],  // Input: 125x130 (8-bit)
    output [7:0] prediction
);

// Clock divider (100MHz → 25MHz)
reg [1:0] clk_div;
wire clk_core = clk_div[1];
always @(posedge clk_100MHz) clk_div <= clk_div + 1;

// Layer 1: Initial Conv2D (3x3, stride 2)
wire [15:0] conv1_out [0:61][0:64][0:15];  // 62x65x16
conv_layer #(
    .INPUT_CH(1),
    .OUTPUT_CH(16),
    .KERNEL_SIZE(3),
    .STRIDE(2)
) conv1 (
    .clk(clk_core),
    .rst(!rst_n),
    .pixel_in(audio_input),
    .feature_map(conv1_out)
);

// Layer 2: DepthwiseConv2D
wire [15:0] depthwise_out [0:61][0:64][0:15];  // 62x65x16
depthwise_conv #(
    .CHANNELS(16),
    .KERNEL_SIZE(3)
) dw_conv (
    .clk(clk_core),
    .features_in(conv1_out),
    .features_out(depthwise_out)
);

// Layer 3: Pointwise Conv2D (1x1)
wire [15:0] pointwise_out [0:61][0:64][0:31];  // 62x65x32
conv_layer #(
    .INPUT_CH(16),
    .OUTPUT_CH(32),
    .KERNEL_SIZE(1),
    .STRIDE(1)
) pw_conv (
    .clk(clk_core),
    .rst(!rst_n),
    .pixel_in(depthwise_out),
    .feature_map(pointwise_out)
);

// Layer 4: MaxPooling2D
wire [15:0] pooled_out [0:30][0:32][0:31];  // 31x32x32
max_pool #(
    .INPUT_WIDTH(62),
    .INPUT_HEIGHT(65),
    .CHANNELS(32)
) pool1 (
    .clk(clk_core),
    .features_in(pointwise_out),
    .features_out(pooled_out)
);

// Layer 5: Conv2D + BatchNorm
wire [15:0] conv2_out [0:30][0:32][0:31];  // 31x32x32
conv_layer #(
    .INPUT_CH(32),
    .OUTPUT_CH(32),
    .KERNEL_SIZE(3),
    .STRIDE(1)
) conv2 (
    .clk(clk_core),
    .rst(!rst_n),
    .pixel_in(pooled_out),
    .feature_map(conv2_out)
);

wire [15:0] bn_out [0:30][0:32][0:31];
batch_norm #(
    .CHANNELS(32),
    .FEATURE_SIZE(32)
) bn1 (
    .clk(clk_core),
    .features_in(conv2_out),
    .features_out(bn_out)
);

// Layer 6: Global Average Pooling
wire [15:0] gap_out [0:31];  // 32 features
global_avg_pool #(
    .INPUT_WIDTH(31),
    .INPUT_HEIGHT(32),
    .CHANNELS(32)
) gap (
    .clk(clk_core),
    .features_in(bn_out),
    .features_out(gap_out)
);

// Layer 7: Dense (32 units)
wire [15:0] dense1_out [0:31];
dense_layer #(
    .INPUT_SIZE(32),
    .OUTPUT_SIZE(32)
) dense1 (
    .clk(clk_core),
    .rst(!rst_n),
    .features_in(gap_out),
    .features_out(dense1_out)
);

// Layer 8: Final Dense (num_labels)
wire [15:0] logits [0:7];  // Assuming num_labels=8
dense_layer #(
    .INPUT_SIZE(32),
    .OUTPUT_SIZE(8)
) dense_final (
    .clk(clk_core),
    .rst(!rst_n),
    .features_in(dense1_out),
    .features_out(logits)
);

// Final prediction
assign prediction = logits[0][15:8];  // MSB of first logit

endmodule