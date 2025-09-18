ncu -o /root/dev/profiling/fused-moe-w8a8-smem-v1$BS -f --kernel-id ::regex:'^(?!.*elementwise).*': --set full python run_moe.py $BS

