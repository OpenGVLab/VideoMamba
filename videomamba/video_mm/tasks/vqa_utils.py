# coding=utf-8

__author__ = 'aagrawal'

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link: 
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
import sys
import re
import json
import logging
logger = logging.getLogger(__name__)


class VQAEval:
    """Modified from https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py

    gt_data_or_path:
        - List(dict): each dict is {
            question_id: int,
            answer: List(str),
            question_type: str,
            answer_type: str,
            ...
            }
        - str: path to file, use json.load() to load it as List(dict)
    pred_data_or_path:
        - List(dict): each dict is {
            question_id: int,
            answer: str
        }
        - str: path to file, use json.load() to load it as List(dict)
    n: #valid float point
    """

    def __init__(self, gt_data_or_path, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                             "couldnt": "couldn't",
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                             "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                             "hed": "he'd", "hed've": "he'd've",
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                             "Id've": "I'd've", "I'dve": "I'd've",
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                             "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                             "mightn'tve": "mightn't've", "mightve": "might've",
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                             "oclock": "o'clock", "oughtnt": "oughtn't",
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                             "shed've": "she'd've", "she'dve": "she'd've",
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                             "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                             "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                             "someone'dve": "someone'd've",
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                             "somethingd've": "something'd've",
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                             "thered": "there'd", "thered've": "there'd've",
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                             "theyd've": "they'd've",
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                             "twas": "'twas", "wasnt": "wasn't",
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                             "whatll": "what'll", "whatre": "what're",
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                             "wheres": "where's", "whereve": "where've",
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                             "whos": "who's", "whove": "who've", "whyll": "why'll",
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                             "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                             "yall'd've": "y'all'd've",
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                             "youd've": "you'd've", "you'dve": "you'd've",
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                          'zero': '0',
                          'one': '1',
                          'two': '2',
                          'three': '3',
                          'four': '4',
                          'five': '5',
                          'six': '6',
                          'seven': '7',
                          'eight': '8',
                          'nine': '9',
                          'ten': '10'
                          }
        self.articles = ['a',
                         'an',
                         'the'
                         ]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

        self.gt_qid2data, self.gt_qid2processed_ans \
            = self.prepare_gt_data(gt_data_or_path)

    def load_data_or_path(self, data_or_path):
        if isinstance(data_or_path, str):
            with open(data_or_path, "r") as f:
                return json.load(f)
        else:
            return data_or_path

    def prepare_gt_data(self, gt_data_or_path):
        gt_data = self.load_data_or_path(gt_data_or_path)
        gt_qid2processed_ans = {}
        for d in gt_data:
            gtAnswers = [ans for ans in d['answer']]
            if len(set(gtAnswers)) > 1:
                gtAnswers = [self.processPunctuation(e) for e in gtAnswers]
            gt_qid2processed_ans[d["question_id"]] = gtAnswers
        gt_qid2data = {e["question_id"]: e for e in gt_data}
        return gt_qid2data, gt_qid2processed_ans

    def init_entry(self):
        print("Cleaned entries for computing new inputs")
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}

    def evaluate(self, pred_data_or_path):
        self.init_entry()
        pred_data = self.load_data_or_path(pred_data_or_path)
        res = {e["question_id"]: e for e in pred_data}
        gts = self.gt_qid2data
        quesIds = list(gts.keys())
        assert len(res) == len(gts), \
            f"expect the same length, but got len(res) {len(res)} == len(gts) {len(gts)}"

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        accAnsType = {}
        print("computing accuracy")
        step = 0
        for quesId in quesIds:
            resAns = res[quesId]['answer']
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []
            gtAnswers = self.gt_qid2processed_ans[quesId]
            # In order to be consistent with ‘human accuracies’, utahine accuracies are averaged over all 10 choose 9 sets of human annotators.
            for idx in range(len(gtAnswers)):
                otherGTAns = gtAnswers[:idx] + gtAnswers[idx + 1:]
                matchingAns = [e for e in otherGTAns if e == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 100 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        print("Done computing accuracy")

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                       outText,
                                       re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy['perQuestionType'] = {
            quesType: round(100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]), self.n) for quesType
            in accQuesType}
        self.accuracy['perAnswerType'] = {
            ansType: round(100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n) for ansType in
            accAnsType}
        # replace "yes/no" as "yes-no" to be compatible with wandb, since "/" is a reserved char
        # self.accuracy['perAnswerType']['yes-no'] = self.accuracy['perAnswerType'].pop('yes/no')

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), int(progress * 100),
                                                          status)
        sys.stdout.write(text)
        sys.stdout.flush()


def eval_simple_qa_acc(gt_data_or_path, pred_data_or_path):
    """For open-ended QA that has a single answer,
    accuracy is computed based on exact match"""
    if isinstance(gt_data_or_path, str):
        gt_data_or_path = json.load(open(gt_data_or_path, "r"))
    if isinstance(pred_data_or_path, str):
        pred_data_or_path = json.load(open(pred_data_or_path, "r"))
    true_or_false_list = []
    gt_qid2answer = {d["question_id"]: d["answer"] for d in gt_data_or_path}
    pred_qid2answer = {d["question_id"]: d["answer"] for d in pred_data_or_path}
    missing_qids = []
    for qid, answer in gt_qid2answer.items():
        if qid in pred_qid2answer:
            true_or_false_list.append(int(pred_qid2answer[qid] == answer))
        else:  # if not exists automatically False
            true_or_false_list.append(0)
            missing_qids.append(qid)
    logger.info(f"Missing predictions for {len(missing_qids)} questions, example[:3] {missing_qids[:3]}")
    acc = round(100. * sum(true_or_false_list) / len(true_or_false_list), 2)
    return acc


def eval_qa_acc(gt_data_or_path, pred_data_or_path, is_vqa=False):
    if is_vqa:
        vqa_evaluator = VQAEval(gt_data_or_path=gt_data_or_path)
        if isinstance(pred_data_or_path, str):
            pred_data_or_path = json.load(open(pred_data_or_path, "r"))
        vqa_evaluator.evaluate(pred_data_or_path)
        vqa_acc = vqa_evaluator.accuracy["perAnswerType"]  # dict
        vqa_acc["overall"] = vqa_evaluator.accuracy["overall"]
        return vqa_acc
    else:
        acc = eval_simple_qa_acc(gt_data_or_path, pred_data_or_path)
        return dict(overall=acc)
