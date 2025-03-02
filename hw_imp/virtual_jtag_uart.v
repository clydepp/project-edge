module virtual_jtag_uart (
    output reg rx_data
);
    // Intel/Altera Virtual JTAG IP
    wire [1:0] ir_out;
    wire tck, tdi, tdo, virtual_state_sdr, virtual_state_udr;
    
    // Instantiate Virtual JTAG IP
    virtual_jtag #(
        .sld_auto_instance_index("YES"),
        .sld_instance_index(0),
        .sld_ir_width(2)
    ) virtual_jtag_inst (
        .tck(tck),                          // JTAG test clock
        .tdi(tdi),                          // JTAG test data input
        .ir_out(ir_out),                    // JTAG instruction register
        .tdo(tdo),                          // JTAG test data output
        .virtual_state_sdr(virtual_state_sdr),  // JTAG shift DR state
        .virtual_state_udr(virtual_state_udr)   // JTAG update DR state
    );
    
    always @(posedge tck) begin
        if (virtual_state_sdr && ir_out == 2'b01) begin
            rx_data <= tdi;  // Directly use TDI as RX data
        end
    end
endmodule