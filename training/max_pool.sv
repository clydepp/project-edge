module max_pool #(
    parameter INPUT_WIDTH = 62,
    parameter INPUT_HEIGHT = 65,
    parameter CHANNELS = 32,
    parameter BIT_WIDTH = 16,
    parameter WEIGHT_FILE = "conv_weights.mem",  
    parameter BIAS_FILE = "conv_bias.mem"        
)(
    input clk,
    input [BIT_WIDTH-1:0] features_in [0:INPUT_HEIGHT-1][0:INPUT_WIDTH-1][0:CHANNELS-1],
    output [BIT_WIDTH-1:0] features_out [0:(INPUT_HEIGHT/2)-1][0:(INPUT_WIDTH/2)-1][0:CHANNELS-1]
);

generate
    genvar c, i, j;
    for (c = 0; c < CHANNELS; c = c + 1) begin: channel
        for (i = 0; i < INPUT_HEIGHT/2; i = i + 1) begin: row
            for (j = 0; j < INPUT_WIDTH/2; j = j + 1) begin: col
                always @(posedge clk) begin
                    // 2x2 window comparison
                    reg [BIT_WIDTH-1:0] val1, val2, val3, val4;
                    reg [BIT_WIDTH-1:0] max1, max2, final_max;
                    
                    val1 = features_in[i*2][j*2][c];
                    val2 = features_in[i*2+1][j*2][c];
                    val3 = features_in[i*2][j*2+1][c];
                    val4 = features_in[i*2+1][j*2+1][c];
                    
                    // Manual max implementation
                    max1 = (val1 > val2) ? val1 : val2;
                    max2 = (val3 > val4) ? val3 : val4;
                    final_max = (max1 > max2) ? max1 : max2;
                    
                    features_out[i][j][c] <= final_max;
                end
            end
        end
    end
endgenerate

endmodule