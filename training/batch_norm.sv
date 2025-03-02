module batch_norm #(
    parameter CHANNELS = 32,
    parameter FEATURE_SIZE = 32,
    parameter BIT_WIDTH = 16,
    parameter FRAC_BITS = 8
)(
    input clk,
    input [BIT_WIDTH-1:0] features_in [0:FEATURE_SIZE-1][0:FEATURE_SIZE-1][0:CHANNELS-1],
    output reg [BIT_WIDTH-1:0] features_out [0:FEATURE_SIZE-1][0:FEATURE_SIZE-1][0:CHANNELS-1]
);

// Batch norm parameters (gamma, beta, mean, variance)
reg [BIT_WIDTH-1:0] gamma [0:CHANNELS-1];
reg [BIT_WIDTH-1:0] beta [0:CHANNELS-1];
reg [BIT_WIDTH-1:0] mean [0:CHANNELS-1];
reg [BIT_WIDTH-1:0] inv_std [0:CHANNELS-1];  // Pre-computed 1/sqrt(var + eps)
initial begin
    $readmemh("gamma.mem", gamma);
    $readmemh("beta.mem", beta);
    $readmemh("mean.mem", mean);
    $readmemh("inv_std.mem", inv_std);  // Pre-calculated inverse std
end

generate
    genvar c, i, j;
    for (c = 0; c < CHANNELS; c = c + 1) begin: channel
        for (i = 0; i < FEATURE_SIZE; i = i + 1) begin: row
            for (j = 0; j < FEATURE_SIZE; j = j + 1) begin: col
                always @(posedge clk) begin
                    // Signed intermediate calculations
                    reg signed [BIT_WIDTH:0] centered;  // BIT_WIDTH+1 bits for subtraction
                    reg signed [BIT_WIDTH*2-1:0] scaled;
                    reg signed [BIT_WIDTH*2-1:0] shifted;
                    
                    // Batch norm: (x - μ) * γ * inv_std + β
                    // Step 1: Center the input
                    centered = features_in[i][j][c] - mean[c];
                    
                    // Step 2: Scale with gamma and inverse std
                    scaled = centered * gamma[c];
                    shifted = scaled * inv_std[c];
                    
                    // Step 3: Add beta and truncate
                    features_out[i][j][c] <= (shifted >>> FRAC_BITS) + beta[c];
                end
            end
        end
    end
endgenerate

endmodule