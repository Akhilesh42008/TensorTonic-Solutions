import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    trained_weights=[w.clone() for w in weights]
    trained_biases=[b.clone() for b in biases]
    loss_history=[]

    def forward():
        activations=[inputs]
        h=inputs
        for w,b in zip(trained_weights,trained_biases):
            h=torch.tanh(h@w.T+b)
            activations.append(h)
        return activations

    for _ in range(steps):
        activations=forward()
        predictions=activations[-1]
        y=targets.reshape_as(predictions)
        loss=torch.sum((predictions-y)**2)
        loss_history.append(loss.clone())

        delta=2*(predictions-y)*(1-predictions**2)
        grad_weights=[None]*len(trained_weights)
        grad_biases=[None]*len(trained_biases)

        for i in range(len(trained_weights)-1,-1,-1):
            grad_weights[i]=delta.T@activations[i]
            grad_biases[i]=delta.sum(dim=0)
            if i>0:
                delta=(delta@trained_weights[i])*(1-activations[i]**2)

        for i in range(len(trained_weights)):
            trained_weights[i]=trained_weights[i]-learning_rate*grad_weights[i]
            trained_biases[i]=trained_biases[i]-learning_rate*grad_biases[i]

    predictions=forward()[-1]
    final_loss=torch.sum((predictions-targets.reshape_as(predictions))**2)
    predictions=predictions.squeeze(1)

    return predictions,final_loss,trained_weights,trained_biases,loss_history
    pass
