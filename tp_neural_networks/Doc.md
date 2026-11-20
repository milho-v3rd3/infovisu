## TP Neural Networks

### First Steps

- To enter the cluster machine : 
```bash
ssh -YK nash.ensimag.fr
```

- We will always be running the code using the following, updated command :

```bash
srun --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 <file.py>
```

This allows us to run the code using the ensimag cluster, as well as choose all of the parameters for this task.

### Evaluation

! To run eval.py and predict.py we must set the weights_only parameter to False when loading, as per PyTorch documentation instrutions.

```python
model = torch.load(model_path, weights_only=False).to(config.DEVICE)
```

And run the following command :

```python
srun --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 eval.py output/last_model.pth
```

### Prediction

We can test it by running the example command :

```python
srun --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 predict.py output/test_data.csv 
```

In order to display only the failed cases, we some treatment to the command line parameters :

```python
for path in sys.argv[1:]:
    if path.endswith('.csv'):
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(path).read().strip().split("\n"):
            # TODO: read bounding box annotations
            filename, _, _, _, _, label = row.split(',')
            filename = os.path.join(config.IMAGES_PATH, label, filename)
            # TODO: add bounding box annotations here
            data.append((filename, None, None, None, None, label))
    elif path == "-fo" or path == "--fail_only":
        display_fails_only = True
    else:
        data.append((path, None, None, None, None, None))
```

### Plotting Training Loss and Accuracy

```python
plt.style.use("ggplot")

    x = np.arange(0,config.NUM_EPOCHS,1)

    training_loss_values = plots["Training loss"]
    training_accuracy_values = plots["Training class accuracy"]
    validation_loss_values = plots["Validation loss"]
    validation_accuracy_values = plots["Validation class accuracy"]

    fig, ax = plt.subplots(2,2)
    fig.suptitle('Training Loss and Accuracy')

    ax[0,0].plot(x,training_loss_values)
    ax[0,0].set_title("Training Loss")
    ax[0,1].plot(x,training_accuracy_values)
    ax[0,1].set_title("Training Accuracy")
    ax[1,0].plot(x,validation_loss_values)  
    ax[1,0].set_title("Validation Loss")
    ax[1,1].plot(x,validation_accuracy_values)
    ax[1,1].set_title("Validation Accuracy")

    plt.show()
```

### Storing best Epoch (Model) in terms of Best Validation Accuracy

```python
 for index, model_accuracy in enumerate(plots['Validation class accuracy']) :
            if model_accuracy > highest_accuracy :
                highest_acc_model = index
            elif model_accuracy == highest_accuracy :
                losses = plots['Validation loss']
                curr_model_loss = losses[index]
                prev_model_loss = losses[index-1]
                if curr_model_loss < prev_model_loss :
                    highest_acc_model = index
                else :
                    highest_acc_model = index - 1 
```

### Shrinking Validation Set and Training Set

- We must alter *cut_val* and *test_val* :

```python
cut_val = int(0.01 * len(data))   
cut_test = int(0.2 * len(data))  
```
- For a cut_val of 1% and a cut_test of 20% :

```bash
*** train set accuracy
	Mean accuracy for all labels: 0.5

	Mean accuracy for label motorcycle: 1.0
		 10 over 10 samples

	Mean accuracy for label airplane: 0.0
		 0 over 7 samples

	Mean accuracy for label face: 0.0
		 0 over 3 samples

*** validation set accuracy
	Mean accuracy for all labels: 0.40932642487046633

	Mean accuracy for label motorcycle: 1.0
		 157 over 157 samples

	Mean accuracy for label airplane: 0.006802721088435374
		 1 over 147 samples

	Mean accuracy for label face: 0.0
		 0 over 82 samples

*** test set accuracy
	Mean accuracy for all labels: 0.389059618930547

	Mean accuracy for label motorcycle: 1.0
		 631 over 631 samples

	Mean accuracy for label airplane: 0.0030959752321981426
		 2 over 646 samples

	Mean accuracy for label face: 0.0
		 0 over 350 samples
```

- For a cut_val of 5% and a cut_test of 20% :

```bash
*** train set accuracy
	Mean accuracy for all labels: 0.9603960396039604

	Mean accuracy for label motorcycle: 1.0
		 42 over 42 samples

	Mean accuracy for label airplane: 0.9534883720930233
		 41 over 43 samples

	Mean accuracy for label face: 0.875
		 14 over 16 samples

*** validation set accuracy
	Mean accuracy for all labels: 0.819672131147541

	Mean accuracy for label motorcycle: 0.928
		 116 over 125 samples

	Mean accuracy for label airplane: 0.8828828828828829
		 98 over 111 samples

	Mean accuracy for label face: 0.5217391304347826
		 36 over 69 samples

*** test set accuracy
	Mean accuracy for all labels: 0.8254456054087277

	Mean accuracy for label motorcycle: 0.9350237717908082
		 590 over 631 samples

	Mean accuracy for label airplane: 0.9024767801857585
		 583 over 646 samples

	Mean accuracy for label face: 0.4857142857142857
		 170 over 350 samples
```

- For a cut_val of 10% and a cut_test of 20% :

```bash
*** train set accuracy
	Mean accuracy for all labels: 0.9852216748768473

	Mean accuracy for label motorcycle: 1.0
		 87 over 87 samples

	Mean accuracy for label airplane: 0.961038961038961
		 74 over 77 samples

	Mean accuracy for label face: 1.0
		 39 over 39 samples

*** validation set accuracy
	Mean accuracy for all labels: 0.9458128078817734

	Mean accuracy for label motorcycle: 1.0
		 80 over 80 samples

	Mean accuracy for label airplane: 0.8831168831168831
		 68 over 77 samples

	Mean accuracy for label face: 0.9565217391304348
		 44 over 46 samples

*** test set accuracy
	Mean accuracy for all labels: 0.9342347879532883

	Mean accuracy for label motorcycle: 0.9857369255150554
		 622 over 631 samples

	Mean accuracy for label airplane: 0.9009287925696594
		 582 over 646 samples

	Mean accuracy for label face: 0.9028571428571428
		 316 over 350 samples
```

### Commenting Batch Normalizations and Dropout

- In the *network.py* file, comment batch normalizations and droupout

Here are the results : 

```bash
*** train set accuracy
	Mean accuracy for all labels: 0.9605911330049262

	Mean accuracy for label motorcycle: 0.9885057471264368
		 86 over 87 samples

	Mean accuracy for label airplane: 0.935064935064935
		 72 over 77 samples

	Mean accuracy for label face: 0.9487179487179487
		 37 over 39 samples

*** validation set accuracy
	Mean accuracy for all labels: 0.9556650246305419

	Mean accuracy for label motorcycle: 1.0
		 80 over 80 samples

	Mean accuracy for label airplane: 0.8831168831168831
		 68 over 77 samples

	Mean accuracy for label face: 1.0
		 46 over 46 samples

*** test set accuracy
	Mean accuracy for all labels: 0.9262446220036877

	Mean accuracy for label motorcycle: 0.9873217115689382
		 623 over 631 samples

	Mean accuracy for label airplane: 0.8947368421052632
		 578 over 646 samples

	Mean accuracy for label face: 0.8742857142857143
		 306 over 350 samples
```

### Train the configuration with no dropout, no batch normalization, 200 epochs, 0.005 for cut_val and 0.10 for cut_test

```bash
*** train set accuracy
	Mean accuracy for all labels: 1.0

	Mean accuracy for label motorcycle: 1.0
		 4 over 4 samples

	Mean accuracy for label airplane: 1.0
		 4 over 4 samples

	Mean accuracy for label face: 1.0
		 2 over 2 samples

*** validation set accuracy
	Mean accuracy for all labels: 0.6580310880829016

	Mean accuracy for label motorcycle: 0.8072289156626506
		 67 over 83 samples

	Mean accuracy for label airplane: 0.5342465753424658
		 39 over 73 samples

	Mean accuracy for label face: 0.5675675675675675
		 21 over 37 samples

*** test set accuracy
	Mean accuracy for all labels: 0.6765027322404371

	Mean accuracy for label motorcycle: 0.7946554149085795
		 565 over 711 samples

	Mean accuracy for label airplane: 0.6224066390041494
		 450 over 723 samples

	Mean accuracy for label face: 0.5631313131313131
		 223 over 396 samples
```