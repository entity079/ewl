from torch import nn


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        # Handle None arguments for optional parameters
        processed_args = []
        for arg in args:
            if arg is not None:
                if isinstance(arg, (tuple, list)):
                    processed_args.append(arg)
                else:
                    processed_args.append((arg,))
            else:
                # For None arguments, create a dummy tuple - assume same length as first non-None arg
                if processed_args and len(processed_args[0]) > 0:
                    processed_args.append((None,) * len(processed_args[0]))
                else:
                    processed_args.append((None,))
        
        if self.weight_factors is None:
            weights = (1, ) * len(processed_args[0])
        else:
            weights = self.weight_factors

        return sum([weights[i] * self.loss(*[arg[i] for arg in processed_args]) for i in range(len(processed_args[0])) if weights[i] != 0.0])
