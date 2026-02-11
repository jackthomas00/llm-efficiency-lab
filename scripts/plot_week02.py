import matplotlib.pyplot as plt

# Median decode_s values (from your 3-run sweep)
tokens = [32, 64, 128, 256]

kv_decode = [1.176, 1.905, 3.744, 7.659]
baseline_decode = [2.369, 4.975, 11.153, 14.236]

# -------- Plot 1: Decode Time --------
plt.figure()
plt.plot(tokens, kv_decode)
plt.plot(tokens, baseline_decode)

plt.xlabel("max_new_tokens")
plt.ylabel("decode_s (seconds)")
plt.title("Decode Time vs max_new_tokens")

plt.legend(["KV Cache", "Baseline"])
plt.savefig("docs/reports/assets/week02_decode_time.png", bbox_inches="tight")
plt.close()


# -------- Plot 2: Tokens Per Second --------
# Approximate TPS from medians (tokens / decode_s)
kv_tps = [t / d for t, d in zip(tokens, kv_decode)]
baseline_tps = [t / d for t, d in zip(tokens, baseline_decode)]

plt.figure()
plt.plot(tokens, kv_tps)
plt.plot(tokens, baseline_tps)

plt.xlabel("max_new_tokens")
plt.ylabel("tokens_per_second")
plt.title("Throughput vs max_new_tokens")

plt.legend(["KV Cache", "Baseline"])
plt.savefig("docs/reports/assets/week02_tps.png", bbox_inches="tight")
plt.close()

print("Plots saved to docs/reports/assets/")
