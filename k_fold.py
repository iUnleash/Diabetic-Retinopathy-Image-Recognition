    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run on gpu if available
    splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(vectorized_data['train_context_array'], vectorized_data['train_context_label_array']))

    # using a numpy array because it's faster than a list
    predictionsfinal = torch.zeros((len(vectorized_data['test_context_array']),1), dtype=torch.float32)
    test_data = torch.tensor(vectorized_data['test_context_array'], dtype=torch.long).to(device)
    test = data_utils.TensorDataset(test_data)
    testloader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    print("--- Running Cross Validation ---")
    # Using K-Fold Cross Validation to train the model and predict the test set by averaging out the predictions across folds
    for i, (train_idx, valid_idx) in enumerate(splits):
        print("\n")
        print("--- Fold Number: {} ---".format(i+1))
        x_train_fold = torch.tensor(vectorized_data['train_context_array'][train_idx], dtype=torch.long).to(device)
        y_train_fold = torch.tensor(vectorized_data['train_context_label_array'][train_idx], dtype=torch.float32).to(device)
        x_val_fold = torch.tensor(vectorized_data['train_context_array'][valid_idx], dtype=torch.long).to(device)
        y_val_fold = torch.tensor(vectorized_data['train_context_label_array'][valid_idx], dtype=torch.float32).to(device)
        
        train = data_utils.TensorDataset(x_train_fold, y_train_fold)
        valid = data_utils.TensorDataset(x_val_fold, y_val_fold)
        
        trainloader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        validloader = data_utils.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
        
        model = Neural_Network(HIDDEN_DIM, embedding_tensor, EMBEDDING_DIM, totalpadlength, vocablen)
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        torch.backends.cudnn.benchmark = True #memory
        torch.backends.cudnn.enabled = True #memory https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/
        f1_list = []
        best_f1 = 0 

        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            iteration = 0
            running_loss = 0.0
            model.train()
            for i, (context, label) in enumerate(trainloader):
                iteration += 1
                # zero out the gradients from the old instance
                optimizer.zero_grad()
                # Run the forward pass and get predicted output
                yhat = model.forward(context) #required dimensions for batching
                # Compute Binary Cross-Entropy
                loss = criterion(yhat, label)
                loss.backward()
                optimizer.step()
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                running_loss += float(loss.item())
    
                if not i%100:
                    print("Epoch: {:03d}/{:03d} | Batch: {:03d}/{:03d} | Cost: {:.4f}".format(
                            epoch+1,NUM_EPOCHS, i+1,len(trainloader),running_loss/iteration))
                    iteration = 0
                    running_loss = 0.0

            # Get the accuracy on the validation set for each epoch
            model.eval()
            with torch.no_grad():
                valid_predictions = torch.zeros((len(x_val_fold),1))
                valid_labels = torch.zeros((len(x_val_fold),1))
                for a, (context, label) in enumerate(validloader):
                    yhat = model.forward(context)
                    valid_predictions[a*BATCH_SIZE:(a+1)*BATCH_SIZE] = (sig_fn(yhat) > 0.5).int()
                    valid_labels[a*BATCH_SIZE:(a+1)*BATCH_SIZE] = label.int()
    
                f1score = f1_score(valid_labels,valid_predictions,average='macro') #not sure if they are using macro or micro in competition
                f1_list.append(f1score)
                
            print('--- Epoch: {} | Validation F1: {} ---'.format(epoch+1, f1_list[-1])) 
            running_loss = 0.0
            
            if f1_list[-1] > best_f1: #save if it improves validation accuracy 
                best_f1 = f1_list[-1]
                torch.save(model.state_dict(), 'train_valid_best.pth') #save best model
                
                
        kfold_test_predictions = torch.zeros((len(vectorized_data['test_context_array']),1))
        
        model.load_state_dict(torch.load('train_valid_best.pth'))
        model.eval()
        with torch.no_grad():
            for a, context in enumerate(testloader):
                yhat = model.forward(context[0])
                kfold_test_predictions[a*BATCH_SIZE:(a+1)*BATCH_SIZE] = sig_fn(yhat)
            
            
            predictionsfinal += (kfold_test_predictions/N_SPLITS)
            
        # removing the file so that the next split can update it
        os.remove("train_valid_best.pth")    
        
    predictionsfinal = (predictionsfinal > THRESHOLD).int()
    output = pd.DataFrame(list(zip(test_ids.tolist(),predictionsfinal.numpy().flatten())))
    output.columns = ['qid', 'prediction']
    print(output.head())
    output.to_csv('submission.csv', index=False)
