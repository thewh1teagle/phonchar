"""
Custom BERT model for phoneme classification
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import torch
from torch import nn
from transformers.utils import ModelOutput
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from src import config
from src.tokenize import CONSONANT_CLASSES, VOWEL_CLASSES, decode, Prediction


@dataclass
class PhonemeLogitsOutput(ModelOutput):
    """Output containing logits for all prediction heads"""
    consonant_logits: torch.FloatTensor = None
    vowel_logits: torch.FloatTensor = None
    stress_logits: torch.FloatTensor = None
    flip_vowel_logits: torch.FloatTensor = None
    
    def detach(self):
        return PhonemeLogitsOutput(
            self.consonant_logits.detach(),
            self.vowel_logits.detach(),
            self.stress_logits.detach(),
            self.flip_vowel_logits.detach()
        )


@dataclass
class PhonemeOutput(ModelOutput):
    """Output containing loss, logits, and hidden states"""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[PhonemeLogitsOutput] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PhonemeLabels(ModelOutput):
    """Labels for all prediction heads"""
    consonant: Optional[torch.LongTensor] = None
    vowel: Optional[torch.LongTensor] = None
    stress: Optional[torch.LongTensor] = None
    flip_vowel: Optional[torch.LongTensor] = None
    
    def detach(self):
        return PhonemeLabels(
            self.consonant.detach(),
            self.vowel.detach(),
            self.stress.detach(),
            self.flip_vowel.detach()
        )
    
    def to(self, device):
        return PhonemeLabels(
            self.consonant.to(device),
            self.vowel.to(device),
            self.stress.to(device),
            self.flip_vowel.to(device)
        )


class PhonemeHead(nn.Module):
    """Multi-head classifier for phoneme prediction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Store class counts
        if not hasattr(config, 'num_consonants'):
            config.num_consonants = len(CONSONANT_CLASSES)
            config.num_vowels = len(VOWEL_CLASSES)
        
        self.num_consonants = config.num_consonants
        self.num_vowels = config.num_vowels
        
        # Create classifiers for each head
        self.consonant_cls = nn.Linear(config.hidden_size, self.num_consonants)
        self.vowel_cls = nn.Linear(config.hidden_size, self.num_vowels)
        self.stress_cls = nn.Linear(config.hidden_size, 2)
        self.flip_vowel_cls = nn.Linear(config.hidden_size, 2)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Forward pass through all classifier heads.
        
        Args:
            hidden_states: BERT output hidden states [batch, seq_len, hidden_size]
            labels: Dict with keys: consonant, vowel, stress, flip_vowel
        
        Returns:
            (loss, logits) tuple
        """
        # Run each classifier on the hidden states
        consonant_logits = self.consonant_cls(hidden_states)
        vowel_logits = self.vowel_cls(hidden_states)
        stress_logits = self.stress_cls(hidden_states)
        flip_vowel_logits = self.flip_vowel_cls(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # Calculate loss for each head (ignoring -100 labels)
            consonant_loss = loss_fct(
                consonant_logits.view(-1, self.num_consonants),
                labels['consonant'].view(-1)
            )
            vowel_loss = loss_fct(
                vowel_logits.view(-1, self.num_vowels),
                labels['vowel'].view(-1)
            )
            stress_loss = loss_fct(
                stress_logits.view(-1, 2),
                labels['stress'].view(-1)
            )
            flip_vowel_loss = loss_fct(
                flip_vowel_logits.view(-1, 2),
                labels['flip_vowel'].view(-1)
            )
            
            # Sum all losses
            loss = consonant_loss + vowel_loss + stress_loss + flip_vowel_loss
        
        logits = PhonemeLogitsOutput(
            consonant_logits=consonant_logits,
            vowel_logits=vowel_logits,
            stress_logits=stress_logits,
            flip_vowel_logits=flip_vowel_logits
        )
        
        return loss, logits


class BertForPhonemeClassification(BertPreTrainedModel):
    """BERT model for phoneme classification with multi-head architecture"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Dropout
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Multi-head phoneme classifier
        self.phoneme_head = PhonemeHead(config)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], PhonemeOutput]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Dict with keys: consonant, vowel, stress, flip_vowel
            ... (other standard BERT arguments)
        
        Returns:
            PhonemeOutput with loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through BERT
        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = bert_outputs[0]
        hidden_states = self.dropout(hidden_states)
        
        # Forward through phoneme head
        loss, logits = self.phoneme_head(hidden_states, labels)
        
        if not return_dict:
            return (loss, logits) + bert_outputs[2:]
        
        return PhonemeOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )
    
    def predict(
        self,
        sentences: List[str],
        tokenizer: BertTokenizerFast,
        padding: str = 'longest',
        preserve_unknown: bool = True
    ) -> List[str]:
        """
        Predict IPA phonemes for input sentences.
        
        Args:
            sentences: List of Hebrew text strings
            tokenizer: DictaBERT tokenizer
            padding: Padding strategy for tokenizer
            preserve_unknown: If True, preserve original chars for non-Hebrew.
                            If False, output Ø for non-Hebrew chars.
        
        Returns:
            List of IPA strings (space-separated phonemes)
        """
        # Tokenize inputs
        inputs = tokenizer(
            sentences,
            padding=padding,
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop('offset_mapping')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.forward(**inputs, return_dict=True)
        
        logits = outputs.logits
        
        # Get argmax predictions for each head
        consonant_preds = logits.consonant_logits.argmax(dim=-1).cpu().tolist()
        vowel_preds = logits.vowel_logits.argmax(dim=-1).cpu().tolist()
        stress_preds = logits.stress_logits.argmax(dim=-1).cpu().tolist()
        flip_vowel_preds = logits.flip_vowel_logits.argmax(dim=-1).cpu().tolist()
        
        # Decode predictions to IPA
        results = []
        for sent_idx, (sentence, sent_offsets) in enumerate(zip(sentences, offset_mapping)):
            # Map token predictions back to character predictions
            char_consonants = []
            char_vowels = []
            char_stresses = []
            char_flip_vowels = []
            
            for token_idx, (start, end) in enumerate(sent_offsets):
                # Skip special tokens
                if start == end:
                    continue
                
                # DictaBERT is character-level
                char_idx = start
                
                if char_idx < len(sentence):
                    char_consonants.append(consonant_preds[sent_idx][token_idx])
                    char_vowels.append(vowel_preds[sent_idx][token_idx])
                    char_stresses.append(stress_preds[sent_idx][token_idx])
                    char_flip_vowels.append(flip_vowel_preds[sent_idx][token_idx])
            
            # Pad predictions to match sentence length
            while len(char_consonants) < len(sentence):
                char_consonants.append(CONSONANT_CLASSES.index(config.NONE))
                char_vowels.append(VOWEL_CLASSES.index(config.NONE))
                char_stresses.append(0)
                char_flip_vowels.append(0)
            
            # Decode to IPA
            preds = Prediction(
                consonant=char_consonants[:len(sentence)],
                vowel=char_vowels[:len(sentence)],
                stress=char_stresses[:len(sentence)],
                flip_vowel=char_flip_vowels[:len(sentence)]
            )
            
            ipa = decode(sentence, preds, preserve_unknown=preserve_unknown)
            results.append(ipa)
        
        return results

