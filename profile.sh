ncu -o /root/dev/profiling/profile$BS -f --kernel-id ::regex:'^(?!.*elementwise).*': --set full python run_moe.py $BS

