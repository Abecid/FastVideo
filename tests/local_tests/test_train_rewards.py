import torch
import pytest

from fastvideo.train.methods.rl.rewards import (
    MultiRewardScorer,
    build_multi_reward_scorer,
    media_to_uint8_array,
    normalize_reward_weights,
    select_first_frame,
)


def test_select_first_frame_for_video_tensor():
    video = torch.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)

    frame = select_first_frame(video)

    assert frame.shape == (2, 3, 5, 6)
    torch.testing.assert_close(frame, video[:, :, 0])


def test_select_first_frame_keeps_frame_tensor():
    frame = torch.randn(2, 3, 5, 6)

    selected = select_first_frame(frame)

    assert selected is frame


def test_multi_reward_weighted_sum_with_injected_scorers():
    def pickscore(media, prompts):
        assert media.shape == (2, 3, 4, 5, 6)
        assert prompts == ["a", "b"]
        return torch.tensor([1.0, 2.0])

    def clipscore(media, prompts):
        assert media.shape == (2, 3, 4, 5, 6)
        assert prompts == ["a", "b"]
        return torch.tensor([0.5, 1.5])

    scorer = MultiRewardScorer(
        {"pickscore": 2.0, "clipscore": 3.0},
        scorers={
            "pickscore": pickscore,
            "clipscore": clipscore,
        },
    )

    scores = scorer(torch.zeros(2, 3, 4, 5, 6), ["a", "b"])

    torch.testing.assert_close(scores["pickscore"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(scores["clipscore"], torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(scores["avg"], torch.tensor([3.5, 8.5]))


def test_multi_reward_validates_score_shape():
    scorer = MultiRewardScorer(
        {"pickscore": 1.0},
        scorers={"pickscore": lambda media, prompts: torch.tensor([[1.0], [2.0]])},
    )

    with pytest.raises(ValueError, match="must return shape"):
        scorer(torch.zeros(2, 3, 4, 5, 6), ["a", "b"])


def test_normalize_reward_weights_accepts_nested_reward_config():
    reward_weights, backend = normalize_reward_weights({
        "backend": "genrl",
        "rewards": {
            "videoalign_vq": 1,
            "videoalign_mq": 0.5,
        },
    })

    assert backend == "genrl"
    assert reward_weights == {
        "videoalign_vq": 1.0,
        "videoalign_mq": 0.5,
    }


def test_build_multi_reward_scorer_accepts_nested_config_with_injected_scorers():
    scorer = build_multi_reward_scorer(
        {
            "backend": "genrl",
            "rewards": {
                "videoalign_vq": 2.0,
                "videoalign_ta": 3.0,
            },
        },
        scorers={
            "videoalign_vq": lambda media, prompts: torch.tensor([1.0, 2.0]),
            "videoalign_ta": lambda media, prompts: torch.tensor([0.5, 1.5]),
        },
    )

    scores = scorer(torch.zeros(2, 3, 4, 5, 6), ["a", "b"])

    torch.testing.assert_close(scores["avg"], torch.tensor([3.5, 8.5]))


def test_media_to_uint8_array_converts_bcthw_video():
    media = torch.ones(2, 3, 4, 5, 6) * 0.5

    arr = media_to_uint8_array(media)

    assert arr.shape == (2, 4, 5, 6, 3)
    assert arr.dtype.name == "uint8"
