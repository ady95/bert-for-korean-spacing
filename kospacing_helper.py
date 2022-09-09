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

    def __init__(self):
        config = OmegaConf.load("config/eval_config.yaml")
        self.preprocessor = Preprocessor(config.max_len)

        self.model = SpacingBertModel(config, None, None, None)
        checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint["state_dict"])
        
        self.config = config

    # 띄어쓰기 처리
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
                print(input_string)

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
        
        print(len(char_list), char_list)
        print(len(slot_labels), slot_labels)

        if len(char_list) != len(slot_labels):
            # 길이가 서로 다른 경우 (주로 …, ㎞ 같은 특수문자를 사용시 발생)
            return None

        for i in range(len(char_list) -1, -1, -1):
            # print(i)
            slot = slot_labels[i]
            if slot == 'S':
                char_list.insert(i +1, ' ')
            elif slot == 'E' and i < len(char_list) -1:
                char_list.insert(i +1, ' ')
        
        return "".join(char_list)

    # GroundTruth와 예측 결과를 비교해서 0~1.0 사이의 점수로 리턴함
    @staticmethod
    def diff_score(gt_text, pred_text):
        offset_gt = 0
        offset_pred = 0
        
        diff_count = 0
        max_len = max(len(gt_text), len(pred_text))
        
        # GT의 전체 띄워쓰기 개수
        gt_spacing_count = len(gt_text) - len(gt_text.replace(" ", ""))
        
        for index in range(max_len):
            index_gt = index + offset_gt
            index_pred = index + offset_pred

            print(index_gt, gt_text)
            print(index_pred, pred_text)
            if gt_text[index_gt] == pred_text[index_pred]:
                pass
            elif gt_text[index_gt] == " ":
                # GT쪽에만 띄어쓰기가 있을때
                diff_count += 1
                offset_pred -= 1
            elif pred_text[index_pred] == " ":
                # PRED쪽에만 띄어쓰기가 있을때
                diff_count += 1
                offset_gt -= 1
                # GT의 전체 띄워쓰기 개수 +1
                gt_spacing_count += 1

        # print("gt_spacing_count:", gt_spacing_count, "diff_count:", diff_count)
        score = 1 - diff_count / gt_spacing_count
        return score
    
    if __name__ == "__main__":
        import os
        from kospacing_helper import KoSpacingHelper
        # gt_text = "아버지가 방에 들어가신다"
        # pred_text = "아버지가방에 들어 가신다"
        ret = KoSpacingHelper.diff_score(gt_text, pred_text)
        helper = KoSpacingHelper()
        ret = helper.predict([
                    "2차세계대전의상처에서헤어나지못했던사람들은인간존재와생의근원을탐구하는그의작품에열광했고,발표하는작품마다독자들의열렬한찬사를받았다.", 
                    "소아암 환아도 우리 이웃, 영화 레터스 투 갓을 현실로"])
        print(ret)

        exit()

        
        from tqdm import tqdm
        import common_util
        
        helper = KoSpacingHelper()
        all_sentence_list = common_util.read_lines("data/test.txt")
        # sentence_list = "그것도 그렇지만 한달간 유럽 다니면서 3번 추락했던 에어2s와 다르게 반년 가까이 동남아를 여행하고 있다."
        # sentence_list = "그것도그렇지만한달간유럽다니면서3번추락했던에어2s와다르게반년가까이동남아를여행하고있지만단한번도추락을한적도없는안정성도비교가되지않을것같기도합니다.일례로커다란비행기가작은비행기보다더안전하다는말도있죠.특별한설명이없어도이해하시리라생각합니다.연결거리도상당히차이가나는것같구요."
        # sentence_list = sentence_list.replace(" ", "")
        # print(len(sentence_list))
        # print(sentence_list[:128])
        # sentence_list = [sentence_list]

        all_sentence_list = all_sentence_list[:100]
        # all_sentence_list = all_sentence_list
        # print(all_sentence_list[:10])
        result_list = helper.predict(all_sentence_list)
        print(len(result_list))
        print(result_list[:10])
        # exit()


        score_count = 0
        total_score = 0
        for start_index in range(0, len(all_sentence_list), 128):
            end_index = min(start_index +128, len(all_sentence_list))
            print(start_index, end_index)
            sentence_list = all_sentence_list[start_index:end_index]
            print(sentence_list)
   
            result_list = helper.predict(sentence_list)
            for i, pred_text in enumerate(tqdm(result_list)):
                if pred_text is None:
                    continue
                gt_text = sentence_list[i]
                score = KoSpacingHelper.diff_score(gt_text, pred_text)
                # print(score, gt_text, pred_text)
                total_score += score
                score_count += 1
        
        average_score = total_score / score_count
        print("average_score:", average_score)

        exit()

        ret = helper.predict(sentence_list)
        # print(ret)

        # 문장별 비교 테스트
        for i, sentence in enumerate(sentence_list):
            result = ret[i]
            check_same = sentence == result
            print(check_same)
            print(sentence)
            print(result)



        

