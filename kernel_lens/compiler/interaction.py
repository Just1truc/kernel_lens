from typing import Optional
from abc import ABC, abstractmethod

class InteractionHandler(ABC):
    """
    Abstract base class for handling interactive queries during compilation.
    This allows the compiler to be used in terminals, GUIs, or fully automated pipelines.
    """
    
    @abstractmethod
    def ask_tensor_kind(self, kernel_name: str, tensor_name: str, shape: tuple) -> str:
        """
        Invoked when the compiler needs to know if a tensor is an input or an output.
        Must return 'input' or 'output'.
        """
        pass


class TerminalInteractionHandler(InteractionHandler):
    """The default handler that uses standard terminal input()."""
    
    def ask_tensor_kind(self, kernel_name: str, tensor_name: str, shape: tuple) -> str:
        while True:
            choice = input(f"[{kernel_name}] Tensor '{tensor_name}' shape={shape}. Is it an [i]nput or [o]utput? ").strip().lower()
            if choice in ['i', 'o']:
                return 'input' if choice == 'i' else 'output'
            print("Invalid choice. Please type 'i' or 'o'.")


class AutoInteractionHandler(InteractionHandler):
    """A handler that attempts to guess based on heuristics, throwing an error if it fails."""
    
    def ask_tensor_kind(self, kernel_name: str, tensor_name: str, shape: tuple) -> str:
        if 'out' in tensor_name.lower():
            print(f"[Auto-Resolve] '{tensor_name}' mapped to OUTPUT.")
            return 'output'
        elif 'in' in tensor_name.lower() or 'ptr' in tensor_name.lower():
             print(f"[Auto-Resolve] '{tensor_name}' mapped to INPUT.")
             return 'input'
        else:
            raise ValueError(f"Auto-resolution failed for tensor '{tensor_name}' in '{kernel_name}'. Please provide explicit mapping.")