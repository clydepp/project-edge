module conv_layer #(
    parameter INPUT_SIZE = 124,
    parameter OUTPUT_SIZE = 62,
    parameter KERNEL_SIZE = 3,
	 parameter OUTPUT_CHANNELS = 16,
    parameter WEIGHT_FILE = "conv_weights.mem",  
    parameter BIAS_FILE = "conv_bias.mem" 
)(
    input clk,
    input rst,
    input [7:0] pixel_in [INPUT_SIZE-1:0][INPUT_SIZE-1:0],
    output reg [15:0] feature_map [OUTPUT_SIZE-1:0][OUTPUT_SIZE-1:0] // Q8.8 fixed-point
);

// BRAM storing quantized weights
reg [7:0] weights [0:KERNEL_SIZE*KERNEL_SIZE-1];
reg [7:0] bias [0:OUTPUT_CHANNELS-1];
initial $readmemh("conv2d_weights.mem", weights);
initial $readmemh("conv2d_bias.mem", bias);

integer acc, i, j, m, n;

// Convolution computation
always @(posedge clk) begin
    if (rst) begin /* Reset logic */ end
    else begin
        for (i=0; i<OUTPUT_SIZE; i = i + 1) begin
            for (j=0; j<OUTPUT_SIZE; j = j + 1) begin
                // Multiply-Accumulate (MAC) operations
                for (m=0; m<KERNEL_SIZE; m = m + 1) begin
                    for (n=0; n<KERNEL_SIZE; n = n + 1) begin
                        acc = acc + $signed(pixel_in[i*2+m][j*2+n]) * $signed(weights[m*KERNEL_SIZE+n]);
                    end
                end
                feature_map[i][j] <= (acc >>> 8) + $signed(bias); // Scale back
            end
        end
    end
end
endmodule