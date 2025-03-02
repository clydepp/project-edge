module relu #(
    parameter WIDTH = 16
)(
    input signed [WIDTH-1:0] data_in,
    output reg signed [WIDTH-1:0] data_out
);
always @(*) begin
    data_out = (data_in[WIDTH-1]) ? 0 : data_in; // ReLU
end
endmodule