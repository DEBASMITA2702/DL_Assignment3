%%writefile AttentionWeightsFetch.py
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn
import Utilities_Device_Trainings
import Utilities_Plotting
import Utilities_Tensor
from Heatmap_Core import createPlot, createAttentionPerCharacter, createHeatMap

def plotAttn(model,
             inputSequence,
             outputSequence,
             vocabulary,
             fontName='/kaggle/input/bengalifont/BengaliFont.ttf'):
    """
    Parameters:
      - model: your encoder-decoder framework
      - inputSequence: LongTensor [B, src_len] or list [src_str]
      - outputSequence: LongTensor [B, tgt_len] or list [tgt_str]
      - vocabulary: PrepareVocabulary instance
      - fontName: path to a Bengali TTF for plotting
    Returns:
      - attention_matrix: numpy array of shape [tgt_len x src_len]
    """

    model.eval()

    ''' Auto-convert from list[str] → LongTensor of indices'''
    if isinstance(inputSequence, list):
        eng_str = inputSequence[0]
        # map characters → indices
        char2idx_en = vocabulary.charToIndexDictForEnglish
        eng_idxs = [char2idx_en[c] for c in eng_str]
        eng_idxs.append(vocabulary.endOfSequenceIndex)
        inputSequence = torch.LongTensor([eng_idxs])

    if isinstance(outputSequence, list):
        bn_str = outputSequence[0]
        char2idx_bn = vocabulary.charToIndexDictForBengali
        bn_idxs = [char2idx_bn[c] for c in bn_str]
        bn_idxs.append(vocabulary.endOfSequenceIndex)
        outputSequence = torch.LongTensor([bn_idxs])

    with torch.no_grad():
        '''transpose & send to device'''
        src = Utilities_Device_Trainings.setDevice(inputSequence.T)
        tgt = Utilities_Device_Trainings.setDevice(outputSequence.T)
        modelEval, attention = model(src, tgt, teacherRatio=0.0)

        ''' collapse batch/time dims'''
        modelEval = Utilities_Tensor.extractColumn(modelEval)
        attention = Utilities_Tensor.extractColumn(attention)
        attn_seq = modelEval.argmax(dim=2)

        ''' reorder to [batch, tgt_len, src_len]'''
        attention = Utilities_Tensor.reorderDimensions(attention, 1, 0, 2)

        ''' transpose inputs back'''
        src = src.T
        attn_seq = attn_seq.T

        ''' prepare plotting grid'''
        _, axes = createPlot()

        attention_matrix = None
        B = src.size(0)
        for row in range(B):
            ''' find true lengths'''
            src_len = (src[row] == vocabulary.endOfSequenceIndex).nonzero(as_tuple=True)
            src_len = src_len[0][0].item()+1 if src_len[0].numel()>0 else src.size(1)

            tgt_seq = attn_seq[row]
            tgt_len = (tgt_seq == vocabulary.endOfSequenceIndex).nonzero(as_tuple=True)
            tgt_len = tgt_len[0][0].item()+1 if tgt_len[0].numel()>0 else tgt_seq.size(0)

            ''' build [tgt_len x src_len] attention-per-char '''
            attentionPerCharacter = createAttentionPerCharacter(
                attention, row, tgt_len, src_len
            )

            '''putting the real attention weights'''
            attention_matrix = attentionPerCharacter.cpu().numpy()

            ''' build tick labels'''
            xTicks, yTicks = Utilities_Plotting.createXandYticks(
                tgt_len, src_len, vocabulary,
                attn_seq, src, row
            )

            '''plot into the grid'''
            if row < len(axes):
                axes = createHeatMap(
                    attentionPerCharacter, axes, row,
                    tgt_len, src_len,
                    xTicks, yTicks,
                    fontName
                )
                
        plt.close()

    return attention_matrix