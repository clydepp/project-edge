module dsp_inference_fpga (

    input  		        			MAX10_CLK1_50,     // 50 MHz clock input from board
    
    // UART Interface
    input  		        			UART_RXD,     // Receive data from USB-Blaster UART
    output 		        			UART_TXD,     // Transmit data to USB-Blaster UART
    
    output reg		  [9:0]		LEDR,          
    
    input  			  [1:0]  	KEY,       // pushbuttons
    output 			  [7:0]  	HEX0,      
	 output		     [7:0]		HEX1,
	 output		     [7:0]		HEX2,
	 output		     [7:0]		HEX3,
	 output		     [7:0]		HEX4,
	 output		     [7:0]		HEX5
);

nios_system u0(
		.clk_clk                           (MAX10_CLK1_50), // clk.clk
		.reset_reset_n                     (1'b1),  // reset.reset_n
		.button_external_connection_export (KEY[1:0]), // button_external_connection.export
		.led_external_connection_export    (LEDR[9:0]),    //    led_external_connection.export
		.hex0_external_connection_export   (HEX0),   //   hex0_external_connection.export
		.hex1_external_connection_export   (HEX1),   //   hex1_external_connection.export
		.hex2_external_connection_export   (HEX2),   //   hex2_external_connection.export
		.hex3_external_connection_export   (HEX3),   //   hex3_external_connection.export
		.hex4_external_connection_export   (HEX4),   //   hex4_external_connection.export
		.hex5_external_connection_export   (HEX5)    //   hex5_external_connection.export
	);

endmodule