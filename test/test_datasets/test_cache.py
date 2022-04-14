"""Tests for cache functionality."""
#  Copyright (c) 2022 Continental Automotive GmbH

import os
import time
from typing import Dict, Tuple, Any

import pytest
import torch
from torch.utils.data import DataLoader

import hybrid_learning.datasets.transforms as trafos
from hybrid_learning.datasets import caching


# pylint: disable=redefined-outer-name
# pylint: disable=not-callable
# pylint: disable=no-self-use


@pytest.fixture
def dict_cache() -> caching.DictCache:
    """A simple DictCache instance."""
    cache = caching.DictCache()
    cache.clear()
    yield cache
    cache.clear()


@pytest.fixture
def file_cache(tmp_path: str) -> caching.PTCache:
    """A simple DictCache instance."""
    cache = caching.PTCache(cache_root=tmp_path)
    cache.clear()
    yield cache
    cache.clear()


@pytest.fixture
def casc_cache(dict_cache, file_cache) -> caching.CacheCascade:
    """A simple CacheCascade."""
    return caching.CacheCascade(dict_cache, file_cache)


def fun(cache: caching.DictCache):
    """Example multiprocess cache operation."""
    for _ in range(10):
        cache.put('a', cache.load('a') + 2)


def gradual_filling(cache: caching.Cache, test_dict: Dict = None):
    """Helper func for the test_gradual_filling methods.

    :param cache: the cache to test
    :param test_dict: a dictionary of ``{cache_descriptor: cache_object}``
        for testing of put and load; defaults to a 4-value dict with differing
        descriptor types.``
    """
    test_dict = test_dict or {1: torch.tensor(1),
                              '2': torch.tensor(2),
                              'c': torch.tensor(3),
                              8: torch.tensor(4)}
    data = DummyDataset(test_dict=test_dict, cache=cache)
    # Check that the example dataset class fills cache gradually:
    # pylint: disable=consider-using-enumerate
    for i in range(len(data)):
        _ = data[i]
        assert len(list(cache.descriptors())) == i + 1, \
            ("Gradual filling failed for cache {};\nCache content:\n{}"
             ).format(str(cache), cache.as_dict())
    cache.clear()


def multiple_workers(cache: caching.Cache, num_items: int = 1000,
                     test_dict: Dict[str, Any] = None):
    """Helper function wrapped by the test_multiple_workers methods.

    :param cache: the cache to test
    :param num_items: if ``test_dict`` is not given, the number of items that
        should be generated for the ``test_dict``
    :param test_dict: a dictionary of ``{cache_descriptor: cache_object}``
        for testing of put and load;
        defaults to ``{str(i): torch.tensor(i) for i in range(num_items)}``
    """
    if test_dict is None:
        test_dict = {str(i): torch.tensor(i) for i in range(num_items)}
    data = DummyDataset(test_dict=test_dict, cache=cache)
    # Use data loader with several workers
    loader = DataLoader(dataset=data, num_workers=2, batch_size=10)
    for epoch in range(2):
        start_time = time.time()
        for idx, data in enumerate(loader):
            if epoch == 0 and idx == 0:
                assert len(list(cache.descriptors())) < len(test_dict)
        end_time = time.time()
        print("\nCache {}, epoch {}: {}s".format(
            cache, epoch, end_time - start_time))
    assert cache.as_dict() == test_dict
    cache.clear()


class DummyDataset(torch.utils.data.Dataset):
    """Test dataset with caching."""

    def __init__(self, test_dict: Dict, cache: caching.Cache):
        """Init."""
        super().__init__()
        self.test_dict = test_dict
        self.cache = cache

    def __getitem__(self, i):
        """Basic getitem including cache."""
        desc = list(self.test_dict.keys())[i]
        item = self.cache.load(desc)
        if item is None:
            item = self.test_dict[desc]
            self.cache.put(desc, item)
        return item

    def __len__(self) -> int:
        """Length."""
        return len(self.test_dict)


class TestDictCaching:
    """Bundled test methods for RAM."""

    def test_put(self, dict_cache: caching.DictCache):
        """Test insertion and overwriting of cache."""
        # Standard put
        test_dict = {"a": 1, 2: "b"}
        for k, content in test_dict.items():
            dict_cache.put(k, content)

        # Test vals are cached
        assert dict(dict_cache.cache) == test_dict

        # Test descriptors()
        assert sorted(dict_cache.descriptors(), key=str) \
               == sorted(test_dict.keys(), key=str)

        # Test load
        for k in test_dict:
            assert dict_cache.load(k) == test_dict[k]

        # Overwriting values:
        key = list(test_dict.keys())[0]
        dict_cache.put(key, 42)
        assert dict_cache.load(key) == 42

        # None raises ValueError
        with pytest.raises(ValueError):
            dict_cache.put("c", None)

    def test_clear(self, dict_cache: caching.DictCache):
        """Test cache clearing."""
        assert len(dict_cache.cache) == 0
        test_dict = {"a": 1, 2: 3.3}
        for k, content in test_dict.items():
            dict_cache.put(k, content)
        assert dict(dict_cache.cache) == test_dict

        dict_cache.clear()
        assert dict(dict_cache.cache) == {}

    def test_repr(self, dict_cache):
        """Test repr()"""
        assert repr(dict_cache) == "DictCache()"

    def test_gradual_filling(self, dict_cache: caching.DictCache):
        """Test whether the caches can be gradually filled by DummyDataset."""
        gradual_filling(dict_cache)

    def test_multiple_workers(self, dict_cache: caching.DictCache):
        """Use caches in data loader with several workers."""
        multiple_workers(dict_cache, 10000)

    # def test_multiprocess_put(self, dict_cache: caching.DictCache):
    #     """Test load and put in a multiprocess setting."""
    #     dict_cache.put(1, '1')
    #     dict_cache.put('a', 2)
    #
    #     proc1 = Process(target=fun, args=(dict_cache,))
    #     proc2 = Process(target=fun, args=(dict_cache,))
    #     proc1.start()
    #     proc2.start()
    #     proc1.join()
    #     proc2.join()
    #
    #     assert dict(dict_cache.cache) == {1: '1', 'a': 2 * 21}


class TestTensorDictCaching(TestDictCaching):
    """Bundled test methods for RAM."""

    @staticmethod
    @pytest.fixture
    def dict_cache() -> caching.TensorDictCache:
        """A simple TensorDictCache instance."""
        cache = caching.TensorDictCache()
        cache.clear()
        yield cache
        cache.clear()

    def test_repr(self, dict_cache: caching.TensorDictCache):
        """Make sure that repr doesn't raise."""
        repr(dict_cache)

    def test_put(self, dict_cache: caching.TensorDictCache):
        """Test insertion and overwriting of cache."""
        # Standard put
        test_dict = {"a": torch.tensor(1), 2: torch.tensor([3.3, 2])}
        for k, content in test_dict.items():
            dict_cache.put(k, content)

        # Test descriptors()
        assert sorted(dict_cache.descriptors(), key=str) \
               == sorted(test_dict.keys(), key=str)

        # Test load
        for k in test_dict:
            assert torch.equal(dict_cache.load(k), test_dict[k])

        # Overwriting values:
        key = list(test_dict.keys())[0]
        dict_cache.put(key, torch.tensor(42))
        assert dict_cache.load(key) == 42

        # None raises ValueError
        with pytest.raises(ValueError):
            dict_cache.put("c", None)

    def test_load(self, dict_cache: caching.TensorDictCache):
        """Test whether all loaded tensors are returned as dense tensors."""
        # Standard put
        test_dict = {"a": torch.tensor(1), 2: torch.tensor([3.3, 2])}
        for k, content in test_dict.items():
            dict_cache.put(k, content)

        # Test load
        for k in test_dict:
            content: torch.Tensor = dict_cache.load(k)
            assert not content.is_sparse, f"sparse cache content leaked: {content}"

    def test_multiple_workers(self, dict_cache: caching.TensorDictCache):
        """Use caches in data loader with several workers."""
        # Only non-sparse tensor caching uses multiprocessing capable dict
        dict_cache = caching.TensorDictCache(sparse=False)
        multiple_workers(dict_cache, 10000)


class TestPTCaching:
    """Bundled test methods for torch pickle file caching."""

    def test_put(self, file_cache):
        """Test insertion and overwriting of cache."""
        # Standard put
        test_dict = {"a": 1, 2: 3.3}
        for k, content in test_dict.items():
            file_cache.put(k, content)

        # test files exist
        test_descriptors = {str(k): v for k, v in test_dict.items()}
        for desc in test_descriptors:
            assert os.path.isfile(os.path.join(file_cache.cache_root,
                                               desc + file_cache.FILE_ENDING))

        # test load
        for desc in test_descriptors:
            assert file_cache.load(desc) == test_descriptors[desc]
        for desc in test_dict:
            assert file_cache.load(desc) == test_dict[desc]

        # test descriptors listed
        assert sorted(file_cache.descriptors(), key=str) == \
               sorted(test_descriptors.keys(), key=str)

        # Overwriting values:
        key = list(test_dict.keys())[0]
        file_cache.put(key, 42)
        assert file_cache.load(key) == 42

        # None raises ValueError
        with pytest.raises(ValueError):
            # noinspection PyTypeChecker
            file_cache.put("c", None)

    def test_put_batch(self, file_cache):
        # Standard batch put
        test_dict = {"a": 1, 2: 3.3}
        file_cache.put_batch(test_dict.keys(), test_dict.values())
        items = {d: file_cache.load(d) for d in file_cache.descriptors()}

        # Should be same as put
        file_cache.clear()
        for k, content in test_dict.items():
            file_cache.put(k, content)
        assert sorted(file_cache.descriptors(), key=str) == sorted(items.keys())
        for k, content in test_dict.items():
            assert file_cache.load(k) == content

        # Overwriting values:
        file_cache.put_batch(test_dict.keys(), test_dict.values())

    @pytest.mark.parametrize('descs,objs', [
        # (['a'], [1, 2]),  # different lengths
        # (['a', 'b'], [1]),  # different lengths
        (['a', 'b'], 1),  # non-iterables
        (5, [1, 2]),  # non-iterables
    ])
    def test_put_batch_raise(self, file_cache, descs, objs):
        with pytest.raises(ValueError):
            file_cache.put_batch(descs, objs)

    @pytest.mark.parametrize('test_dict,load_descs,return_none_if,return_is_none', [
        ({"a": 1, 2: 3.3}, ['a', 2], 'any', False),
        ({"a": 1, 2: 3.3}, ['a'], 'any', False),
        ({"a": 1, 2: 3.3}, [2], 'any', False),
        ({"a": 1, 2: 3.3}, [2, 'b'], 'any', True),
        ({"a": 1, 2: 3.3}, ['c', 'b'], 'any', True),
        ({"a": 1, 2: 3.3}, ['c', 'b'], 'all', True),
        ({"a": 1, 2: 3.3}, ['a', 'b'], 'all', False),
        ({"a": 1, 2: 3.3}, ['a', 'b'], 'never', False),
        ({"a": 1, 2: 3.3}, ['c', 'b'], 'never', False),
    ])
    def test_load_batch(self, file_cache, test_dict, return_none_if, load_descs, return_is_none):
        file_cache.put_batch(test_dict.keys(), test_dict.values())
        load = file_cache.load_batch(load_descs, return_none_if=return_none_if)
        if return_is_none:
            assert load is None
        else:
            assert list(load) == list(test_dict.get(k, None) for k in load_descs)

    def test_clear(self, file_cache):
        """Test cache clearing."""
        assert len(list(file_cache.descriptors())) == 0
        test_dict = {"a": 1, 2: 3.3}
        for k, content in test_dict.items():
            file_cache.put(k, content)

        assert file_cache.as_dict() == \
               {str(k): v for k, v in test_dict.items()}

        file_cache.clear()
        assert list(file_cache.descriptors()) == []
        assert os.listdir(file_cache.cache_root) == []

    def test_repr(self, file_cache):
        """Test repr()"""
        repres: str = repr(file_cache)
        for expr in ("PTCache", str(file_cache.cache_root),
                     str(file_cache.device),
                     repr(file_cache.before_put)):
            assert expr in repres

    def test_gradual_filling(self, file_cache):
        """Test whether the caches can be gradually filled by DummyDataset."""
        gradual_filling(file_cache)

    def test_multiple_workers(self, file_cache):
        """Use caches in data loader with several workers."""
        multiple_workers(file_cache, 1000)


class TestNPYCaching(TestPTCaching):
    """Bundled test methods for numpy pickle file caching."""

    @staticmethod
    @pytest.fixture
    def file_cache(tmp_path: str) -> caching.NPYCache:
        """A simple DictCache instance."""
        cache = caching.NPYCache(cache_root=tmp_path)
        cache.clear()
        yield cache
        cache.clear()

    def test_repr(self, file_cache):
        """Test repr()"""
        assert repr(file_cache) == \
               "NPYCache(cache_root={})".format(file_cache.cache_root)


class TestNPZCaching(TestPTCaching):
    """Bundled test methods for numpy pickle file caching."""

    @staticmethod
    @pytest.fixture
    def file_cache(tmp_path: str) -> caching.NPZCache:
        """A simple DictCache instance."""
        cache = caching.NPZCache(cache_root=tmp_path)
        cache.clear()
        yield cache
        cache.clear()

    def test_repr(self, file_cache):
        """Test repr()"""
        assert repr(file_cache) == \
               "NPZCache(cache_root={})".format(file_cache.cache_root)


class TestJPGCache:  # pylint: disable=invalid-name
    """Test for JPGCache."""

    @staticmethod
    @pytest.fixture
    def file_cache(tmp_path: str) -> caching.JPGCache:
        """A simple TensorDictCache instance."""
        cache = caching.JPGCache(cache_root=tmp_path)
        cache.clear()
        yield cache
        cache.clear()

    @staticmethod
    def half_gray_float_image(size: Tuple[int, ...],
                              dtype: torch.dtype = torch.float) -> torch.Tensor:
        """A gray image represented by a torch tensor of ``dtype``.
        The maximum value is 0.4, and 2 pixel values are featured."""
        return torch.cat([
            torch.ones((*size[:-1], size[-1] - 1), dtype=dtype) * 0.4,
            torch.ones((*size[:-1], 1), dtype=dtype) * 0.25],
            dim=-1)

    @pytest.mark.parametrize("size,inp_dtype,mode", [
        ((3, 10, 15), torch.float, 'RGB'),
        ((1, 10, 15), torch.float, 'F'),
        ((1, 10, 15), torch.float, 'L'),
        ((10, 15), torch.float, 'F'),
        ((10, 15), torch.float, 'L'),
        ((3, 10, 15), torch.int, 'RGB'),
        ((1, 10, 15), torch.int, 'F'),
        ((1, 10, 15), torch.int, 'L'),
        ((10, 15), torch.int, 'F'),
        ((10, 15), torch.int, 'L'),
    ])
    def test_put(self, file_cache: caching.JPGCache,
                 size, inp_dtype, mode):
        """Test putting of different channel numbers for float and int
        dtypes."""
        file_cache.mode = mode
        descriptor: str = f"size={size},dtype={inp_dtype},mode={mode})"
        # Example: a white image (not respecting value scaling)
        put: torch.Tensor = torch.ones(size)

        file_cache.put(descriptor, put)
        assert descriptor in file_cache.descriptors()
        loaded: torch.Tensor = file_cache.load(descriptor)
        assert isinstance(loaded, torch.Tensor)

    @pytest.mark.parametrize("size,inp_dtype,mode", [
        ((3, 10, 15), torch.float, 'RGB'),
        ((3, 10, 15), torch.float16, 'RGB'),
        # ((3, 10, 15), torch.bfloat16, 'RGB'),  # no .allclose defined here
        ((3, 10, 15), torch.float32, 'RGB'),
        ((3, 10, 15), torch.float64, 'RGB'),
        ((10, 15), torch.float16, 'L'),
        ((10, 15), torch.float32, 'L'),
        ((10, 15), torch.float64, 'L'),
    ])
    def test_idempotence_autoscaling_float(
            self, file_cache: caching.JPGCache, size, inp_dtype, mode):
        """Test putting of different channel numbers for float dtypes."""
        file_cache.mode = mode
        file_cache.after_load = trafos.ToTensor(dtype=inp_dtype)
        descriptor: str = f"size={size},dtype={inp_dtype},mode={mode})"
        put: torch.Tensor = self.half_gray_float_image(size, inp_dtype)

        file_cache.put(descriptor, put)
        loaded: torch.Tensor = file_cache.load(descriptor)
        if len(put.size()) == 2:
            assert loaded.size()[0] == 1
            loaded = loaded.squeeze(0)
        assert put.dtype == loaded.dtype
        assert put.size() == loaded.size(), f"Size differs for {descriptor}: " \
                                            f"put.size(): {put.size()}; " \
                                            f"loaded.size(): {loaded.size()}"
        assert torch.allclose(put, loaded, atol=3 / 255), \
            f"change for caching {descriptor}:\nput: {put}\nloaded: {loaded}"

    @pytest.mark.parametrize("size,inp_dtype,mode", [
        ((1, 10, 15), torch.float32, 'F'),
        ((10, 15), torch.float32, 'F'),
    ])
    def test_idempotence_nonautoscaling_float(
            self, file_cache: caching.JPGCache, size, inp_dtype, mode):
        """Test putting of different channel numbers for float dtypes."""
        file_cache.mode = mode
        file_cache.before_put = trafos.RecursiveLambda(lambda x: x * 255)
        file_cache.after_load = (trafos.ToTensor()
                                 + trafos.RecursiveLambda(lambda x: x / 255)
                                 + trafos.ToTensor(dtype=inp_dtype))
        descriptor: str = f"size={size},dtype={inp_dtype},mode={mode})"
        put: torch.Tensor = self.half_gray_float_image(size, inp_dtype)

        file_cache.put(descriptor, put)
        loaded: torch.Tensor = file_cache.load(descriptor)
        if len(put.size()) == 2:
            assert loaded.size()[0] == 1
            loaded = loaded.squeeze(0)
        assert put.dtype == loaded.dtype
        assert put.size() == loaded.size(), f"Size differs for {descriptor}: " \
                                            f"put.size(): {put.size()}; " \
                                            f"loaded.size(): {loaded.size()}"
        assert torch.allclose(put, loaded, atol=3 / 255), \
            f"change for caching {descriptor}:\nput: {put}\nloaded: {loaded}"

    @pytest.mark.parametrize("size,inp_dtype,mode", [
        ((3, 10, 15), torch.int, 'RGB'),
        ((3, 10, 15), torch.int8, 'RGB'),
        ((3, 10, 15), torch.uint8, 'RGB'),
        ((3, 10, 15), torch.int16, 'RGB'),
        ((3, 10, 15), torch.int32, 'RGB'),
        ((3, 10, 15), torch.int8, None),
        ((1, 10, 15), torch.int8, 'L'),
        ((1, 10, 15), torch.uint8, 'L'),
        ((10, 15), torch.int8, 'L'),
        ((10, 15), torch.uint8, 'L'),
    ])
    def test_idempotence_autoscalingload_int(
            self, file_cache: caching.JPGCache, size, inp_dtype, mode):
        """Test putting of different channel numbers."""
        file_cache.mode = mode
        file_cache.after_load = (trafos.ToTensor()
                                 + trafos.RecursiveLambda(lambda x: x * 255)
                                 + trafos.ToTensor(dtype=inp_dtype))

        descriptor: str = f"size={size},dtype={inp_dtype},mode={mode})"
        put: torch.Tensor = (self.half_gray_float_image(size) * 255
                             ).to(inp_dtype)
        file_cache.put(descriptor, put)
        loaded: torch.Tensor = file_cache.load(descriptor)
        if len(put.size()) == 2:
            assert loaded.size()[0] == 1
            loaded = loaded.squeeze(0)

        assert inp_dtype == loaded.dtype
        assert put.size() == loaded.size()
        assert torch.allclose(put.to(inp_dtype), loaded, atol=3), \
            f"put:\n{put}\nloaded:\n{loaded}"

    @pytest.mark.parametrize("size,inp_dtype,mode", [
        ((3, 10, 15), torch.bool, 'RGB'),
        ((1, 10, 15), torch.bool, 'L'),
        ((10, 15), torch.bool, 'L'),
    ])
    def test_idempotence_bool(self, file_cache: caching.JPGCache,
                              size, inp_dtype, mode):
        """Test putting of different channel numbers."""
        file_cache.mode = mode
        file_cache.after_load = trafos.ToTensor(dtype=inp_dtype)
        descriptor: str = f"size={size},dtype={inp_dtype},mode={mode})"
        # Example: white mask
        put: torch.Tensor = torch.ones(size, dtype=torch.bool)

        file_cache.put(descriptor, put)
        loaded: torch.Tensor = file_cache.load(descriptor)

        assert inp_dtype == loaded.dtype
        assert put.size()[-2:] == loaded.size()[-2:]
        assert torch.all(torch.eq(put, loaded)), \
            f"put: {put}\nloaded:\n{loaded}"

    @pytest.mark.parametrize("size,inp_dtype,mode", [
        ((1, 10, 15), torch.int16, 'I;16'),
        ((1, 10, 15), torch.int32, 'I'),
        ((10, 15), torch.int16, 'I;16'),
        ((10, 15), torch.int32, 'I'),
    ])
    def test_idempotence_nonautoscaling_int(
            self, file_cache: caching.JPGCache, size, inp_dtype, mode):
        """Test putting for modes ``'I*'``."""
        file_cache.mode = mode
        file_cache.after_load = trafos.ToTensor(dtype=inp_dtype)
        descriptor: str = f"size={size},dtype={inp_dtype},mode={mode})"
        # Example: gray image
        put: torch.Tensor = \
            (torch.ones(size, dtype=torch.uint8) * 120).to(inp_dtype)

        file_cache.put(descriptor, put)
        loaded: torch.Tensor = file_cache.load(descriptor)

        if len(put.size()) == 2:
            assert loaded.size()[0] == 1
            loaded = loaded.squeeze(0)
        assert inp_dtype == loaded.dtype
        assert put.size() == loaded.size()
        assert torch.allclose(put.to(inp_dtype), loaded), \
            f"put: {put}\nloaded: {loaded}"


class TestCacheCascade:
    """Test the CacheCascade caching functionality."""

    def test_init_val_check(self,
                            file_cache,
                            dict_cache: caching.DictCache):
        """Check whether ValueError is raised for invalid init args."""

        # no children caches:
        with pytest.raises(ValueError):
            caching.CacheCascade()

        # wrong sync_by vals:
        with pytest.raises(ValueError):
            caching.CacheCascade(file_cache, dict_cache, sync_by='blub')

    def test_insert_append_remove(self, file_cache,
                                  dict_cache: caching.DictCache):
        """Test insert(), append() and remove()."""
        casc_cache: caching.CacheCascade = caching.CacheCascade(dict_cache)
        assert casc_cache.caches == [dict_cache]

        # append
        assert casc_cache.append(file_cache) is casc_cache
        assert casc_cache.caches == [dict_cache, file_cache]

        # insert
        assert casc_cache.insert(0, file_cache) is casc_cache
        assert casc_cache.caches == [file_cache, dict_cache, file_cache]

        # remove
        assert casc_cache.remove(1) is casc_cache
        assert casc_cache.caches == [file_cache, file_cache]

    def test_put(self, dict_cache: caching.DictCache,
                 file_cache):
        """Test insertion and overwriting of cache."""
        casc_cache: caching.CacheCascade = \
            caching.CacheCascade(dict_cache, file_cache)
        # Standard put
        test_dict = {"a": 1, "2": 3.3}
        for k, content in test_dict.items():
            casc_cache.put(k, content)

        # Test vals are cached
        for k, content in test_dict.items():
            assert casc_cache.load(k) == test_dict[k]
            assert dict_cache.load(k) == test_dict[k]
            assert file_cache.load(k) == test_dict[k]

        # Test descriptors()
        assert sorted(casc_cache.descriptors()) == sorted(test_dict.keys())

        # Overwriting values:
        key = list(test_dict.keys())[0]
        casc_cache.put(key, 42)
        assert casc_cache.load(key) == 42
        assert dict_cache.load(key) == 42
        assert file_cache.load(key) == 42

        # None raises ValueError
        with pytest.raises(ValueError):
            casc_cache.put("c", None)

    def test_clear(self, casc_cache: caching.CacheCascade):
        """Test cache clearing."""
        assert len(list(casc_cache.descriptors())) == 0
        test_dict = {"a": 1, '2': 3.3}
        for k, content in test_dict.items():
            casc_cache.put(k, content)
        for k, content in test_dict.items():
            assert casc_cache.load(k) == test_dict[k]
        assert sorted(casc_cache.descriptors(), key=str) == \
               sorted(test_dict.keys(), key=str)

        casc_cache.clear()
        assert casc_cache.as_dict() == {}
        for cache in casc_cache.caches:
            assert cache.as_dict() == {}

    def test_sync(self, dict_cache, file_cache):
        """Test different sync modes during load."""

        # sync_by='precedence': load from second, update first
        casc_cache: caching.CacheCascade = \
            caching.CacheCascade(dict_cache, file_cache, sync_by='precedence')
        file_cache.put("b", 2)
        assert casc_cache.load("b") == 2
        assert dict_cache.as_dict() == {"b": 2}
        assert file_cache.as_dict() == {"b": 2}
        casc_cache.clear()

        # sync_by='precedence': load from first without updating second
        casc_cache: caching.CacheCascade = \
            caching.CacheCascade(dict_cache, file_cache, sync_by='precedence')
        dict_cache.put("a", 1)
        file_cache.put("a", 2)
        assert casc_cache.load("a") == 1
        assert dict_cache.as_dict() == {"a": 1}
        assert file_cache.as_dict() == {"a": 2}
        casc_cache.clear()

        # sync_by='all': load from first, update second
        casc_cache: caching.CacheCascade = \
            caching.CacheCascade(dict_cache, file_cache, sync_by='all')
        dict_cache.put("a", 1)
        file_cache.put("a", 2)
        assert casc_cache.load("a") == 1
        assert dict_cache.as_dict() == {"a": 1}
        assert file_cache.as_dict() == {"a": 1}
        casc_cache.clear()

        # sync_by='none': load from second without updating first
        casc_cache: caching.CacheCascade = \
            caching.CacheCascade(dict_cache, file_cache, sync_by='none')
        assert casc_cache.sync_by is False
        file_cache.put("b", 2)
        assert casc_cache.load("b") == 2
        assert dict_cache.as_dict() == {}
        assert file_cache.as_dict() == {"b": 2}
        casc_cache.clear()

    def test_repr(self, casc_cache: caching.CacheCascade):
        """Test repr()"""
        assert repr(casc_cache) == "CacheCascade({}, {})".format(
            repr(casc_cache.caches[0]), repr(casc_cache.caches[1])
        )

    def test_gradual_filling(self, casc_cache: caching.CacheCascade):
        """Test whether the caches can be gradually filled by DummyDataset."""
        gradual_filling(casc_cache, test_dict={'1': torch.tensor(1),
                                               '2': torch.tensor(2),
                                               'c': torch.tensor(3),
                                               '8': torch.tensor(4)})

    def test_multiple_workers(self, casc_cache: caching.CacheCascade):
        """Use caches in data loader with several workers."""
        multiple_workers(casc_cache, 1000)


class TestCacheTuple:
    """Test the CacheTuple caching functionality."""

    def test_init_val_check(self,
                            file_cache,
                            dict_cache: caching.DictCache):
        """Check whether ValueError is raised for invalid init args."""
        # fine inits:
        caching.CacheTuple(file_cache, dict_cache, return_none_if=1)
        caching.CacheTuple(file_cache, dict_cache, return_none_if=-1)
        caching.CacheTuple(file_cache, dict_cache, return_none_if=0)

        # no children caches:
        with pytest.raises(ValueError):
            caching.CacheTuple()

        # wrong sync_by vals:
        with pytest.raises(ValueError):
            caching.CacheTuple(file_cache, dict_cache, return_none_if='blub')

    def test_put(self, dict_cache: caching.DictCache,
                 file_cache):
        """Test insertion and overwriting of cache."""
        tup_cache: caching.CacheTuple = \
            caching.CacheTuple(dict_cache, file_cache)
        # Standard put
        test_dict = {"a": (1, 3), "2": (5, 7)}
        for k, content in test_dict.items():
            tup_cache.put(k, content)

        # Test vals are cached
        for k, content in test_dict.items():
            assert tup_cache.load(k) == test_dict[k]
            assert dict_cache.load(k) == test_dict[k][0]
            assert file_cache.load(k) == test_dict[k][1]

        # Test descriptors()
        assert sorted(tup_cache.descriptors()) == sorted(test_dict.keys())

        # Overwriting values:
        key = list(test_dict.keys())[0]
        tup_cache.put(key, (42, 7))
        assert tup_cache.load(key) == (42, 7)
        assert dict_cache.load(key) == 42
        assert file_cache.load(key) == 7

        # Non-subscriptable value raises TypeError
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            tup_cache.put("c", 1)
        # Too short tuple raises TypeError
        with pytest.raises(IndexError):
            tup_cache.put("c", (1,))
        # None value raises ValueError
        with pytest.raises(ValueError):
            tup_cache.put("c", (1, None))

    def test_clear(self, file_cache,
                   dict_cache: caching.DictCache):
        """Test cache clearing."""
        tup_cache: caching.CacheTuple = caching.CacheTuple(dict_cache,
                                                           file_cache)
        assert len(list(tup_cache.descriptors())) == 0
        test_dict = {"a": (1, 3), '2': (5, 7)}
        for k, content in test_dict.items():
            tup_cache.put(k, content)
        for k, content in test_dict.items():
            assert tup_cache.load(k) == test_dict[k]
        assert sorted(tup_cache.descriptors(), key=str) == \
               sorted(test_dict.keys(), key=str)

        tup_cache.clear()
        assert tup_cache.as_dict() == {}
        for cache in tup_cache.caches:
            assert cache.as_dict() == {}

    def test_load_none(self, dict_cache: caching.DictCache,
                       file_cache):
        """Test different sync modes during load."""

        # Gradual filling with (2, 3) should result in:
        gradual_non_none_results: Dict[str, Tuple] = {
            'any': (None, None, (3, 2)),
            'all': (None, (None, 2), (3, 2)),
            'never': ((None, None), (None, 2), (3, 2))
        }

        for return_none_if, fill_results in gradual_non_none_results.items():
            tup_cache: caching.CacheTuple = caching.CacheTuple(
                dict_cache, file_cache, return_none_if=return_none_if)
            assert tup_cache.load("b") == fill_results[0]
            file_cache.put("b", 2)
            assert tup_cache.load("b") == fill_results[1]
            dict_cache.put("b", 3)
            assert tup_cache.load("b") == fill_results[2]
            tup_cache.clear()

    def test_repr(self, file_cache,
                  dict_cache: caching.DictCache):
        """Test repr()"""
        tup_cache: caching.CacheTuple = \
            caching.CacheTuple(dict_cache, file_cache)
        assert repr(tup_cache) == "CacheTuple({}, {})".format(
            repr(dict_cache), repr(file_cache))

        tup_cache: caching.CacheTuple = \
            caching.CacheTuple(dict_cache, file_cache, return_none_if='never')
        assert repr(tup_cache) == \
               ("CacheTuple({}, {}, return_none_if=-1)"
                ).format(repr(dict_cache), repr(file_cache))

    def test_gradual_filling(self, file_cache,
                             dict_cache: caching.DictCache):
        """Test whether the caches can be gradually filled by DummyDataset."""
        gradual_filling(
            caching.CacheTuple(dict_cache, file_cache),
            test_dict={'1': (torch.tensor(1), torch.tensor(1)),
                       '2': (torch.tensor(2), torch.tensor(2)),
                       'c': (torch.tensor(3), torch.tensor(3)),
                       '8': (torch.tensor(4), torch.tensor(4))})

    def test_multiple_workers(self, file_cache,
                              dict_cache: caching.DictCache):
        """Use caches in data loader with several workers."""
        multiple_workers(
            caching.CacheTuple(dict_cache, file_cache),
            test_dict={str(i): (torch.tensor(i), torch.tensor(i))
                       for i in range(1000)}
        )


class TestCacheDict:

    @pytest.fixture
    def cache_dict(self, tmp_path) -> caching.CacheDict:
        """A standard CacheDict."""
        cache = caching.CacheDict({'a': caching.PTCache(cache_root=str(tmp_path)),
                                   'b': caching.DictCache()})
        yield cache
        cache.clear()

    def test_put(self, cache_dict: caching.CacheDict):
        """Test put."""
        test_dict: Dict[Any, Dict] = {
            'first': {'a': 1, 'b': 'blub'},
            'second': {'a': 2, 'b': 'bla'}, }
        cache_dict.put_batch(test_dict.keys(), test_dict.values())
        assert sorted(cache_dict.descriptors()) == sorted(test_dict.keys())

        out_dicts = cache_dict.load_batch(test_dict.keys())
        for desc, out_dict in zip(test_dict.keys(), out_dicts):
            assert out_dict == test_dict[desc]


def test_add(file_cache, dict_cache: caching.Cache):
    """Test concatenation of caches."""
    # Standard add
    caches: caching.Cache = file_cache + dict_cache
    assert isinstance(caches, caching.CacheCascade)
    assert caches.caches == [file_cache, dict_cache]

    # Addition of NoCache and None
    assert file_cache == file_cache + caching.NoCache()
    assert file_cache == caching.NoCache() + file_cache
    assert file_cache == file_cache + None
    assert file_cache == (None + file_cache)

    # Raising
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        _ = file_cache + 1
        # noinspection PyTypeChecker
        _ = 1 + file_cache
