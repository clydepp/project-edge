module global_avg_pool(
    input [15:0] features_in [0:61][0:63],
    output reg [15:0] features_out [0:31]
);

integer ch;

always @(*) begin
    for (ch=0; ch<32; ch = ch + 1) begin
        features_out[ch] = average_all_pixels(features_in, ch);
    end
end
endmodule