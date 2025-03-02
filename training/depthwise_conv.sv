module depthwise_conv #(
    parameter CHANNELS = 16,       // Same as input channels
    parameter KERNEL_SIZE = 3,
    parameter BIT_WIDTH = 16,
    parameter WEIGHT_FILE = "conv_weights.mem",  
    parameter BIAS_FILE = "conv_bias.mem" 
)(
    input clk,
    input [BIT_WIDTH-1:0] features_in [0:61][0:64][0:CHANNELS-1],
    output [BIT_WIDTH-1:0] features_out [0:61][0:64][0:CHANNELS-1]
);

// 3x3 kernel weights for each channel
reg [BIT_WIDTH-1:0] kernel [0:2][0:2][0:CHANNELS-1];
initial begin
    $readmemh("depthwise_weights.mem", kernel);  // Load trained weights
end

generate
    genvar c, i, j;
    for (c = 0; c < CHANNELS; c = c + 1) begin: channel
        for (i = 0; i < 62; i = i + 1) begin: row
            for (j = 0; j < 65; j = j + 1) begin: col
                // Padding handling (same padding)
                wire [9:0] sum;
                always @(posedge clk) begin
                    sum = 0;
                    for (int ki = 0; ki < 3; ki = ki + 1) begin
                        for (int kj = 0; kj < 3; kj = kj + 1) begin
                            if ((i*1 + ki) >= 1 && (i*1 + ki) < 63 && 
                                (j*1 + kj) >= 1 && (j*1 + kj) < 66) begin
                                sum += features_in[i + ki - 1][j + kj - 1][c] * kernel[ki][kj][c];
                            end
                        end
                    end
                    features_out[i][j][c] <= sum[BIT_WIDTH-1:0];
                end
            end
        end
    end
endgenerate

endmodule