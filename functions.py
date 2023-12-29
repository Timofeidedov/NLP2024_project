def get_sentences(text):
    doc = nlp(text)
    sents = []
    positions = []
    for sent in doc.sentences:
        words = []
        word_positions = []
        for word in sent.words:
            words.append(word.lemma)
            word_positions.append((word.start_char, word.end_char))
        sents.append(words)
        positions.append(word_positions)
    return sents, positions

def my_padding(samples):
        ids_tensors = [s[1] for s in samples]
        ids_tensors = pad_sequence(ids_tensors, batch_first=True)

        tags_tensors = [s[2] for s in samples]
        tags_tensors = pad_sequence(tags_tensors, batch_first=True)

        masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1)

        return ids_tensors, tags_tensors, masks_tensors

def train_abte(model, ds, epochs, device='cpu', batch_size=8, lr=1e-5):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=my_padding)

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        num_training_steps = epochs * len(loader)

        losses = []

        all_data = len(loader)-1
        for epoch in range(epochs):
            finish_data = 0
            n_batches = int(len(ds)/batch_size)

            for nb in range((n_batches)):
                ids_tensors, tags_tensors, masks_tensors = next(iter(loader))
                ids_tensor = ids_tensors.to(device)
                tags_tensor = tags_tensors.to(device)
                masks_tensor = masks_tensors.to(device)
                loss = model(ids_tensors=ids_tensor, tags_tensors=tags_tensor, masks_tensors=masks_tensor)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                finish_data += 1
                print("epoch: {}\tbatch: {}/{}\tloss: {}".format(epoch, finish_data, all_data, loss.item()))
            torch.save(model.state_dict(), f'/content/abtebert_epoch{epoch}.pkl')

def predict_abte(model, tokenizer, tokens, device='cpu'):
        word_pieces = [el for token in tokens for el in tokenizer.tokenize(token)]
        ids = tokenizer.convert_tokens_to_ids(word_pieces)
        input_tensor = torch.tensor([ids]).to(device)

        with torch.no_grad():
            outputs = model(input_tensor, None, None)
            _, predictions = torch.max(outputs, dim=2)

        predictions = predictions[0].tolist()
        return predictions

def sent_from_tokens(tokens):
    sent = ''
    for token in tokens:
        if (sent == '' or sent[-1] in '.!?') and token != '':
            if len(token) > 1:
                token = token[0].upper() + token[1:]
            else:
                token = token[0].upper()
        if sent != '' and sent[-1] in '.,:;!?"()':
            sent += ' '
        sent += token
    return sent

def predict_sentiment(tokens, model):
    sent = sent_from_tokens(tokens)
    return model.predict([sent])[0]['label'].lower()

def predict_absa(text_id, text, tokenizer, abte_model, ton_model, device='cpu'):
    aspects = []
    sents, positions = get_sentences(text)
    for i, sent in enumerate(sents):
        aspect_preds = predict_abte(abte_model, tokenizer, sent, device)
        for j, aspect in enumerate(aspect_preds):
            if aspect == 1:
                try:
                    new_asp = {}
                    new_asp['id'] = text_id
                    new_asp['cat'] = classify(sents[i][j])
                    new_asp['token'] = sents[i][j]
                    new_asp['from'] = positions[i][j][0]
                    new_asp['to'] = positions[i][j][1]
                    new_asp['sentiment'] = predict_sentiment(sent, ton_model)
                    aspects.append(new_asp)
                except:
                    pass
    return aspects

def absa_pred_df(reviews, tokenizer, abte_model, ton_model, device='cpu'):
    aspects = []
    for review in tqdm(reviews.iterrows()):
        text_id = review[1]['id']
        text = review[1]['review']
        aspects.extend(predict_absa(text_id, text, tokenizer, abte_model, ton_model, device))
    return pd.DataFrame(aspects)