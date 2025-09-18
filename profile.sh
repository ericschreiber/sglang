ncu -o /root/dev/profiling/profile-prefetch-fp8$BS -f --kernel-id ::regex:'^(?!.*elementwise).*': --set full python run_moe.py $BS

