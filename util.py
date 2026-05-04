import torch

def update_confusion_matrix(conf_matrix, outputs, labels):
    predictions = outputs.argmax(dim=1)

    labels = labels.long()
    predictions = predictions.long()

    for true_label, predicted_label in zip(labels, predictions):
        conf_matrix[true_label, predicted_label] += 1

    return conf_matrix

def evaluate_model(model, data_loader, num_classes, device):
    model.eval()
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            conf_matrix = update_confusion_matrix(
                conf_matrix,
                outputs.detach().cpu(),
                labels.detach().cpu()
            )

    return conf_matrix