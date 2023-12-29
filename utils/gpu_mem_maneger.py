import torch
import gc

class GPUMemoryManager:
    def __init__(self, device_num):
        # Initialising GPU memory tracking variables
        self.device_num = device_num
        self.max_reserved = 0
        self.max_allocated = 0

    @staticmethod
    def byte2MB(bt):
        return round(bt / (1024 ** 2), 3)

    def reset_peak_memory_stats(self):
        torch.cuda.reset_peak_memory_stats()

    def clear_cache(self):
        """Clears the cache by emptying the CUDA cache and running garbage collection"""
        gc.collect()
        torch.cuda.empty_cache()

    def print_memory_stats(self):
        total_memory = torch.cuda.get_device_properties(self.device_num).total_memory
        reserved_memory = torch.cuda.memory_reserved(self.device_num)
        allocated_memory = torch.cuda.memory_allocated(self.device_num)
        free_memory = reserved_memory - allocated_memory

        print('GPU Memory Status:')
        print(f'Total Memory: {self.byte2MB(total_memory)} MB')
        print(f'Reserved Memory: {self.byte2MB(reserved_memory)} MB')
        print(f'Allocated Memory: {self.byte2MB(allocated_memory)} MB')
        print(f'Free Memory: {self.byte2MB(free_memory)} MB')
