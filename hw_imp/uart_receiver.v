module uart_receiver (
    input clk,           // Board clock
    input reset_n,       // Reset signal
    input rx_data,       // UART RX data (connect to JTAG pin)
    output reg [7:0] data_out,  // Received byte
    output reg data_valid       // Pulses high when new byte received
);

    // Parameters for 115200 baud (assuming 50MHz clock)
    parameter CLKS_PER_BIT = 434;  // 50,000,000 / 115,200 ≈ 434
    
    // State machine states
    parameter IDLE = 2'b00;
    parameter START_BIT = 2'b01;
    parameter DATA_BITS = 2'b10;
    parameter STOP_BIT = 2'b11;
    
    reg [1:0] state = IDLE;
    reg [9:0] clk_counter = 0;
    reg [2:0] bit_index = 0;
    
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            data_out <= 8'h00;
            data_valid <= 1'b0;
            clk_counter <= 0;
            bit_index <= 0;
        end else begin
            case (state)
                IDLE: begin
                    data_valid <= 1'b0;
                    clk_counter <= 0;
                    bit_index <= 0;
                    
                    if (rx_data == 1'b0)  // Start bit detected
                        state <= START_BIT;
                end
                
                START_BIT: begin
                    if (clk_counter == CLKS_PER_BIT/2) begin
                        // Check middle of start bit is still low
                        if (rx_data == 1'b0) begin
                            clk_counter <= 0;
                            state <= DATA_BITS;
                        end else
                            state <= IDLE;
                    end else
                        clk_counter <= clk_counter + 1;
                end
                
                DATA_BITS: begin
                    if (clk_counter < CLKS_PER_BIT)
                        clk_counter <= clk_counter + 1;
                    else begin
                        clk_counter <= 0;
                        // Sample the data bit
                        data_out[bit_index] <= rx_data;
                        
                        if (bit_index < 7) begin
                            bit_index <= bit_index + 1;
                        end else begin
                            bit_index <= 0;
                            state <= STOP_BIT;
                        end
                    end
                end
                
                STOP_BIT: begin
                    if (clk_counter < CLKS_PER_BIT)
                        clk_counter <= clk_counter + 1;
                    else begin
                        data_valid <= 1'b1;
                        clk_counter <= 0;
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end
endmodule