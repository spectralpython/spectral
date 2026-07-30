'''Tests of the multilayer perceptron classifier.

Training tests seed `numpy.random` before constructing each `Perceptron`
(whose layer weights are otherwise randomly initialized) so that training
behavior is reproducible; the AND gate is used as training data since it's
linearly separable and converges quickly and reliably.

The demo functions at the bottom of `perceptron.py` (`test_xor`, `test_and`,
etc., and the `if __name__ == '__main__'` block) are standalone scripts
that just forward to `Perceptron.train`, already covered thoroughly below;
they're intentionally not exercised here.

To run the unit tests, type the following from the system command line:

    # pytest spectral/tests/test_perceptron.py
'''

import io

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spectral.algorithms.perceptron import (Perceptron, PerceptronLayer,
                                            and_data)

AND_XY = list(zip(*and_data))


class TestPerceptronLayer:

    def test_shape(self):
        layer = PerceptronLayer((3, 2))
        assert layer.shape == (2, 4)
        assert layer.weights.shape == (2, 4)

    def test_explicit_weights(self):
        weights = np.arange(8, dtype=float).reshape(2, 4)
        layer = PerceptronLayer((3, 2), weights=weights)
        assert_allclose(layer.weights, weights)

    def test_explicit_weights_wrong_shape_raises(self):
        weights = np.zeros((2, 3))
        with pytest.raises(Exception):
            PerceptronLayer((3, 2), weights=weights)

    def test_randomize_weights_unit_length(self):
        layer = PerceptronLayer((5, 4))
        norms = np.sqrt((layer.weights[:, 1:] ** 2).sum(axis=1))
        assert_allclose(norms, np.ones(4))

    def test_input_and_activation(self):
        layer = PerceptronLayer((2, 1))
        layer.weights = np.array([[0.0, 1.0, -1.0]])
        y = layer.input([2.0, 5.0])
        expected = 1. / (1. + np.exp(3.0))
        assert_allclose(y, [expected])

    def test_input_clip(self):
        layer = PerceptronLayer((2, 1))
        layer.weights = np.array([[100.0, 1.0, -1.0]])  # saturates near 1
        y = layer.input([100.0, 0.0], clip=0.1)
        assert_allclose(y, [0.9])

    def test_dy_da(self):
        layer = PerceptronLayer((2, 1))
        layer.weights = np.array([[0.0, 1.0, -1.0]])
        layer.input([2.0, 5.0])
        expected = layer.k * layer.y * (1.0 - layer.y)
        assert_allclose(layer.dy_da(), expected)


class TestPerceptronConstruction:

    def test_invalid_layers_type_raises(self):
        with pytest.raises(Exception):
            Perceptron('not a list')

    def test_too_few_layers_raises(self):
        with pytest.raises(Exception):
            Perceptron([3])

    def test_layer_shapes(self):
        p = Perceptron([3, 4, 2])
        assert p.layers[0].shape == (4, 4)
        assert p.layers[1].shape == (2, 5)


class TestPerceptronInputClassify:

    @pytest.fixture
    def scaled_perceptron(self):
        '''A single-layer Perceptron with known weights and scaling set
        directly (bypassing `train`, which is what normally establishes
        `_scale`/`_offset`) so `input`/`classify` can be checked in
        isolation against a hand-computed expected value.'''
        p = Perceptron([2, 1])
        p._scale = np.array([1.0, 1.0])
        p._offset = np.array([0.0, 0.0])
        p.layers[0].weights = np.array([[0.0, 1.0, -1.0]])
        return p

    def test_input(self, scaled_perceptron):
        result = scaled_perceptron.input([2.0, 5.0])
        expected = 1. / (1. + np.exp(3.0))
        assert_allclose(result, [expected])

    def test_classify(self, scaled_perceptron):
        # Large positive z -> sigmoid saturates near 1 -> rounds to 1.
        assert scaled_perceptron.classify([10.0, 0.0]) == [1]
        # Large negative z -> sigmoid saturates near 0 -> rounds to 0.
        assert scaled_perceptron.classify([0.0, 10.0]) == [0]


class TestSetScaling:

    def test_handles_constant_feature(self):
        '''A feature that's constant across all training samples has zero
        range; `_set_scaling` should avoid a divide-by-zero for it rather
        than producing an inf/nan scale factor.'''
        np.random.seed(0)
        p = Perceptron([2, 1])
        X = [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        p._set_scaling(X)
        assert np.all(np.isfinite(p._scale))


class TestPerceptronTrain:

    def test_converges_on_and_gate(self):
        np.random.seed(0)
        p = Perceptron([2, 1])
        out = io.StringIO()
        trained = p.train(*AND_XY, max_iterations=2000, stdout=out)
        assert trained is True
        assert p.accuracy == 100.0

    @pytest.mark.parametrize('batch', [0, 1, 3])
    def test_batch_options_converge(self, batch):
        np.random.seed(0)
        p = Perceptron([2, 1])
        out = io.StringIO()
        trained = p.train(*AND_XY, max_iterations=3000, batch=batch,
                          stdout=out)
        assert trained is True

    def test_momentum_converges(self):
        np.random.seed(0)
        p = Perceptron([2, 1])
        out = io.StringIO()
        trained = p.train(*AND_XY, max_iterations=3000, momentum=0.5,
                          stdout=out)
        assert trained is True

    def test_accuracy_over_100_runs_full_iterations(self):
        '''Setting `accuracy` above 100 (the maximum achievable) forces
        `train` to run for the full `max_iterations` rather than stopping
        early once 100% accuracy is reached.'''
        np.random.seed(0)
        p = Perceptron([2, 1])
        out = io.StringIO()
        trained = p.train(*AND_XY, max_iterations=5, accuracy=200.0,
                          stdout=out)
        assert trained is False

    def test_stdout_none_is_suppressed(self):
        np.random.seed(0)
        p = Perceptron([2, 1])
        p.train(*AND_XY, max_iterations=5, stdout=None)

    def test_on_iteration_callback_stops_training(self):
        np.random.seed(0)
        p = Perceptron([2, 1])
        calls = []

        def on_iteration(perceptron):
            calls.append(perceptron.accuracy)
            return True

        out = io.StringIO()
        trained = p.train(*AND_XY, max_iterations=100,
                          on_iteration=on_iteration, stdout=out)
        assert trained is True
        assert len(calls) == 1

    def test_keyboard_interrupt_during_training(self):
        np.random.seed(0)
        p = Perceptron([2, 1])

        def raise_interrupt(perceptron):
            raise KeyboardInterrupt

        out = io.StringIO()
        trained = p.train(*AND_XY, max_iterations=100,
                          on_iteration=raise_interrupt, stdout=out)
        assert trained is False
        assert 'KeyboardInterrupt' in out.getvalue()


class RaisingArray(np.ndarray):
    '''An ndarray whose in-place addition always raises KeyboardInterrupt,
    used to simulate a CTRL-C occurring mid weight-update.'''
    def __iadd__(self, other):
        raise KeyboardInterrupt


class TestAdjustWeightsInterrupt:
    '''Tests `Perceptron._adjust_weights`'s own KeyboardInterrupt handling
    directly, isolated from the full `train` loop.'''

    def _make_perceptron_with_raising_weights(self):
        np.random.seed(0)
        p = Perceptron([2, 1])
        layer = p.layers[0]
        layer.dW = np.ones_like(layer.weights) * 0.1
        original_weights = np.array(layer.weights)
        layer.weights = layer.weights.view(RaisingArray)
        return (p, layer, original_weights)

    def test_restores_previous_weights_when_caching(self):
        (p, layer, original_weights) = \
            self._make_perceptron_with_raising_weights()
        out = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            p._adjust_weights(0.3, 0.0, 1, out)
        assert_allclose(np.asarray(layer.weights), original_weights)
        assert_allclose(layer.dW, 0)
        assert 'Restoring previous weights' in out.getvalue()

    def test_leaves_weights_when_caching_disabled(self):
        (p, layer, original_weights) = \
            self._make_perceptron_with_raising_weights()
        p.cache_weights = False
        out = io.StringIO()
        with pytest.raises(KeyboardInterrupt):
            p._adjust_weights(0.3, 0.0, 1, out)
        assert 'cacheing was disabled' in out.getvalue()
