import unittest
from peft import PeftModel
from src.models.lora_model import setup_lora_model


class TestLoraModel(unittest.TestCase):
    def test_returns_peft_model(self):
        model, tokenizer = setup_lora_model(r=2)
        self.assertIsInstance(model, PeftModel)
        self.assertIsNotNone(tokenizer)

    def test_only_lora_params_trainable(self):
        model, _ = setup_lora_model(r=4)
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        
        self.assertTrue(len(trainable_names) > 0)
        for name in trainable_names:
            self.assertTrue("lora_" in name.lower() or "classifier" in name.lower())

    def test_rank_affects_param_count(self):
        model_r2, _ = setup_lora_model(r=2)
        model_r16, _ = setup_lora_model(r=16)
        
        trainable_r2 = sum(p.numel() for p in model_r2.parameters() if p.requires_grad)
        trainable_r16 = sum(p.numel() for p in model_r16.parameters() if p.requires_grad)
        
        self.assertLess(trainable_r2, trainable_r16)

    def test_b_init_zero(self):
        model, _ = setup_lora_model(r=4)
        b_params = [p for name, p in model.named_parameters() if "lora_b" in name.lower()]
        self.assertTrue(len(b_params) > 0)
        for p in b_params:
            self.assertTrue((p == 0).all().item())


if __name__ == "__main__":
    unittest.main()
