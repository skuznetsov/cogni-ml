#!/usr/bin/env crystal

# Empirical route table from Raspberry Pi 5 Q6 tied-head probes.
# CPU is the probe's scalar row-dot oracle over prepacked Q6 rows; V3D includes
# one compute dispatch and mapped-output top1 scan in the probe shape.
# Small-count rows use the warmed cached-prepack sweep from
# scripts/rpi5_q6_allowed_threshold.sh; wider rows are retained from the earlier
# cached full-head indexed sweep.

points = [
  {3, 0.180, 0.105},
  {8, 0.186, 0.299},
  {13, 0.189, 0.475},
  {16, 0.191, 0.604},
  {32, 0.328, 1.188},
  {64, 0.380, 2.383},
  {128, 0.610, 4.815},
  {256, 0.922, 9.642},
  {1024, 2.734, 38.846},
  {4096, 8.621, 155.674},
  {8192, 12.865, 309.318},
  {16384, 22.124, 621.220},
]

puts "allowed\tv3d_ms\tcpu_ms\twinner\tspeedup"
points.each do |count, v3d, cpu|
  if v3d < cpu
    puts "#{count}\t#{v3d}\t#{cpu}\tV3D\t#{(cpu / v3d).round(3)}x"
  else
    puts "#{count}\t#{v3d}\t#{cpu}\tCPU\t#{(v3d / cpu).round(3)}x"
  end
end

first_v3d = points.find { |_, v3d, cpu| v3d < cpu }
if first_v3d
  puts "policy_threshold_min_v3d_allowed=#{first_v3d[0]}"
  puts "recommended_QWEN35_ALLOWED_HEAD_CPU_MAX=#{first_v3d[0] - 1}"
else
  puts "policy_threshold_min_v3d_allowed=none"
  puts "recommended_QWEN35_ALLOWED_HEAD_CPU_MAX=disabled"
end
