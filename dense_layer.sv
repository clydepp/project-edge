module dense_layer #(
    parameter INPUT_SIZE = 32,
    parameter OUTPUT_SIZE = 8,
    parameter WEIGHT_WIDTH = 16,  // Q8.8 format
    parameter ACC_WIDTH = 32,      // Accumulator width
    parameter WEIGHT_FILE = "conv_weights.mem",  
    parameter BIAS_FILE = "conv_bias.mem" 
)(
    input clk,
    input rst,
    input [WEIGHT_WIDTH-1:0] features_in [0:INPUT_SIZE-1],
    output reg [WEIGHT_WIDTH-1:0] features_out [0:OUTPUT_SIZE-1]
);

// Memory declarations
reg [WEIGHT_WIDTH-1:0] weights [0:INPUT_SIZE*OUTPUT_SIZE-1];
reg [WEIGHT_WIDTH-1:0] biases [0:OUTPUT_SIZE-1];

initial begin
    $readmemh("dense_weights.mem", weights);
    $readmemh("dense_biases.mem", biases);
end

always @(posedge clk) begin
    if (rst) begin
        // Reset outputs
        for (int n = 0; n < OUTPUT_SIZE; n++) begin
            features_out[n] <= '0;
        end
    end
    else begin
        // Parallel processing for each neuron
        for (int neuron = 0; neuron < OUTPUT_SIZE; neuron++) begin
            // Signed accumulator with protection against overflow
            logic signed [ACC_WIDTH-1:0] acc = 0;
            
            // Dot product calculation
            for (int i = 0; i < INPUT_SIZE; i++) begin
                automatic int index = neuron * INPUT_SIZE + i;
                acc += $signed(features_in[i]) * $signed(weights[index]);
            end
            
            // Apply bias and quantization
            features_out[neuron] <= $signed(acc[ACC_WIDTH-1:8]) + $signed(biases[neuron]);
        end
    end
end
endmodule