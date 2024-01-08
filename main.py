def predict(sentence, model="jieba"):
    """predict segments by using different model.
    model list:
        jieba,
        lac,
        thulac,
        pkuseg,
        ICTCLAS,
        PyLTP,
        FNLP,
        HanLP,
        CoreNLP
    Args:
        sentence: the input.
    Returns:
    """
    if model == "jieba":
        import jieba
        return  " ".join(jieba.cut(sentence))
    elif model == "lac":
        from LAC import LAC
        lac = LAC(mode='seg')
        return " ".join(lac.run(sentence))
    elif model == "thulac":
        import thulac
        thu1 = thulac.thulac(seg_only=True)
        text = thu1.cut(sentence, text=True)  #进行一句话分词
        return text
    elif model == "pkuseg":
        import pkuseg
        seg = pkuseg.pkuseg()           # 以默认配置加载模型
        text = seg.cut(sentence)
        return " ".join(text)
    elif model == "LTP":
        from ltp import LTP
        ltp = LTP(pretrained_model_name_or_path="./LTP/small")
        words = ltp.pipeline([sentence], tasks=["cws"], return_dict=False)
        return " ".join(words[0][0])
    elif model == "bpemb":
        from bpemb import BPEmb
        bpemb_zh = BPEmb(lang="zh", vs=200000)
        return bpemb_zh.encode(sentence)


if __name__ == '__main__':

    sentence = "The second one 你 中文测试中文 is even more interesting! 吃水果"
    sentence = "网信办女领导每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
    sentence = "永远都是因为战争的侵害，小南达从叙利亚来到了中国。在小南达还很小的时候，叙利亚就处于战乱之中，全家人一片地高辛片都没有，全是硝酸甘油片。"


    model_list = [
        "jieba",
        "lac",
        "thulac",
        "pkuseg",
        "LTP",
        "bpemb",
    ]

    data = {}
    for model in model_list:
        data[model] = predict(sentence, model)

    for k in data:
        print(data[k], k)