import torch
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from preprocessor import Preprocessor
from dataset_for_test import CorpusDataset
from net import SpacingBertModel

# def get_dataloader(data_path, transform, batch_size):
#     dataset = CorpusDataset(data_path, transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size)

#     return dataloader

class KoSpacingHelper:

    def __init__(self, config):
        self.preprocessor = Preprocessor(config.max_len)

        self.model = SpacingBertModel(config, None, None, None)
        checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint["state_dict"])
        
        self.config = config


    def predict(self, sentence_list):
        dataset = CorpusDataset(sentence_list, self.preprocessor.get_input_features)
        test_dataloader = DataLoader(dataset, batch_size=self.config.eval_batch_size)
        self.dataloader = test_dataloader

        result_list = []

        for batch in self.dataloader:
            # input_ids, slot_labels, attention_mask, token_type_ids = item
            input_ids, attention_mask, token_type_ids, slot_labels = batch
 
            outputs = self.model.forward(input_ids, attention_mask, token_type_ids)
            gt_slot_labels, pred_slot_labels = self.model._convert_ids_to_labels(
                outputs, slot_labels
            )

            for i in range(len(input_ids)):
                # print(pred_slot_labels)
                input_token = self.preprocessor.tokenizer.convert_ids_to_tokens(input_ids[i])
                # print(input_token)
                # input_string = helper.preprocessor.tokenizer.convert_tokens_to_string(input_tokens)
                input_string = self.get_text(input_token)
                # print(input_string)

                result_sentence = self.spacing_sentence(input_string, pred_slot_labels[i])
                result_list.append(result_sentence)
            
        return result_list



    def get_text(self, tokens):
        char_list = []
        for token in tokens:
            if token.startswith("["):
                continue
            token = token.replace(self.preprocessor.tokenizer.SPIECE_UNDERLINE, "").strip()
            char_list.append(token)
        
        return "".join(char_list)


    def spacing_sentence(self, text, slot_labels):
        # 1. 빈칸 제거
        char_list = []
        for i, c in enumerate(text):
            if c != ' ':
                char_list.append(c)
        
        # print(len(char_list), char_list)
        # print(len(slot_labels), slot_labels)
        for i in range(len(char_list) -1, -1, -1):
            # print(i)
            slot = slot_labels[i]
            if slot == 'S':
                char_list.insert(i +1, ' ')
            elif slot == 'E' and i < len(char_list):
                char_list.insert(i +1, ' ')
        
        return "".join(char_list)

    
    if __name__ == "__main__":
        import common_util
        from kospacing_helper import KoSpacingHelper
        config = OmegaConf.load("config/eval_config.yaml")
        
        sentence_list = common_util.read_lines(config.test_data_path)
        
        helper = KoSpacingHelper(config)
        ret = helper.predict(sentence_list)
        print(ret)

        exit()
        for batch in helper.dataloader:
            # input_ids, slot_labels, attention_mask, token_type_ids = item
            input_ids, attention_mask, token_type_ids, slot_labels = batch
 
            outputs = helper.model.forward(input_ids, attention_mask, token_type_ids)
            gt_slot_labels, pred_slot_labels = helper.model._convert_ids_to_labels(
                outputs, slot_labels
            )
            # print(gt_slot_labels)
            
            # print(input_ids)
            # print(input_labels)
            for i in range(len(input_ids)):
                # print(pred_slot_labels)
                input_token = helper.preprocessor.tokenizer.convert_ids_to_tokens(input_ids[i])
                # print(input_token)
                # input_string = helper.preprocessor.tokenizer.convert_tokens_to_string(input_tokens)
                input_string = helper.get_text(input_token)
                print(input_string)

                result_sentence = helper.spacing_sentence(input_string, pred_slot_labels[i])
                print(result_sentence)


        

