from pyctcdecode import build_ctcdecoder
class CTCDecoder():
    def __init__(self) -> None:
        pass

    def load_decoder(self,processor,dec_path):
        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        print(sorted_vocab_dict)
        if len(dec_path)!=0:
            decoder = build_ctcdecoder(
                labels=list(sorted_vocab_dict.keys()),
                kenlm_model_path=dec_path
            )
        else:
            decoder = build_ctcdecoder(
                labels=list(sorted_vocab_dict.keys())
            )
        return decoder