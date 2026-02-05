// Each Coyote project needs a vfpga_top.svh file, which is a simple SystemVerilog header
// that provides the interface from/to the Coyote shell. If not provided, the synthesis
// process of Coyote will fail. In this case, the vfpga_top.svh simply instantiates the model_wrapper

// Model wrapper; note the suffix _hls_ip which must be added for HLS kernels in Coyote.
// More details can be found in Example 2 of the Coyote repository.
model_wrapper_hls_ip inst_model(
    // hls-fpga-machine-learning insert axi connections
    .ap_clk               (aclk),
    .ap_rst_n             (aresetn)
);

// Tie-off unused signals to avoid synthesis problems
// hls-fpga-machine-learning tie off host streams
always_comb sq_rd.tie_off_m();
always_comb sq_wr.tie_off_m();
always_comb cq_rd.tie_off_s();
always_comb cq_wr.tie_off_s();
always_comb notify.tie_off_m();
always_comb axi_ctrl.tie_off_s();

logic [63:0] cnt;
always_ff @(posedge aclk) begin
   if (aresetn == 1'b0) begin
     cnt <= '0;
   end else begin
      cnt <= cnt + 1;
   end
end

ila_perf_host inst_ila_perf_host (
    .clk(aclk),
    .probe0(cnt),
    // hls-fpga-machine-learning insert ila
);
