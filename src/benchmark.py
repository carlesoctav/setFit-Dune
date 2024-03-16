from pathlib import Path
from time import perf_counter
from setfit import SetFitModel
from datasets import load_dataset
import evaluate
import numpy as np
import torch
from tqdm.auto import tqdm
import unicodedata
import re

#copied from https://huggingface.co/docs/setfit/tutorials/onnx
class PerformanceBenchmark:
    def __init__(self, model, optim_type):
        self.model = model
        self.optim_type = optim_type

    def compute_size(self):
        state_dict = self.model.model_body.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        tmp_path.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_model(self, query="that loves its characters and communicates something rather beautiful about human nature"):
        latencies = []
        # Warmup
        for _ in range(10):
            _ = self.model([query])
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.model([query])
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(rf"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_model())
        return metrics


if __name__ == "__main__":     
    model = SetFitModel.from_pretrained("carlesoctav/SentimentClassifierBarbieDune-8shot")
    pb = PerformanceBenchmark(model=model, optim_type="bge-small (PyTorch)")
    perf_metrics = pb.run_benchmark()