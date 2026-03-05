from collections import defaultdict

import numpy as np


class ActionsSplitter:
    def __call__(self, actions):
        raise NotImplementedError


class GlobalTemporalSplit(ActionsSplitter):
    def __init__(
        self,
        max_test_users: int = 4096,
        random_seed: int = 31337,
        test_timestamps_fraction: float = 0.1,
        val_timestamp_fraction: float = 0.1,
    ):
        self.max_test_users = max_test_users
        self.random_seed = random_seed
        self.test_timestamps_fraction = test_timestamps_fraction
        self.val_timestamp_fraction = val_timestamp_fraction

    def __call__(self, actions):
        sorted_actions = sorted(actions, key=lambda x: x.timestamp)
        border_timestamp = sorted_actions[
            int(len(sorted_actions) * (1 - self.test_timestamps_fraction))
        ].timestamp

        sorted_actions_before_border_ts = [
            action for action in sorted_actions if action.timestamp < border_timestamp
        ]
        val_border_timestamp = sorted_actions[
            int(len(sorted_actions_before_border_ts) * (1 - self.val_timestamp_fraction))
        ].timestamp

        users_train = defaultdict(list)
        users_test = defaultdict(list)
        eligible_users_test = set()

        for action in sorted_actions:
            if action.timestamp < border_timestamp:
                users_train[action.user_id].append(action)
            elif action.user_id in users_train.keys():
                users_test[action.user_id].append(action)
                # add to the eligible_users_test the ones having at least one positive interaction after the test border timestamp
                # including users having only negative interactions after the test border timestamp makes the ndcg NaN
                if action.rating > 0:
                    eligible_users_test.add(action.user_id)

        train = []
        test = []

        eligible_users_train = set()
        for user in users_train:
            positive_actions_before_val_ts = [
                action
                for action in users_train[user]
                if action.timestamp < val_border_timestamp and action.rating > 0
            ]
            if len(positive_actions_before_val_ts) > 1:
                # add to the eligible_users_train the ones having at least two actions before the validation border timestamp
                eligible_users_train.add(user)

        eligible_users = eligible_users_train.intersection(eligible_users_test)

        np.random.seed(self.random_seed)
        if len(eligible_users) < self.max_test_users:
            test_user_ids = set(eligible_users)
        else:
            eligible_users = list(eligible_users)
            test_user_ids = set(
                np.random.choice(eligible_users, self.max_test_users, replace=False)
            )

        for user_id in eligible_users:
            if user_id in test_user_ids:
                train += users_train[user_id]
                test += users_test[user_id]
            else:
                train += users_train[user_id]

        return (
            sorted(train, key=lambda x: x.timestamp),
            sorted(test, key=lambda x: x.timestamp),
            val_border_timestamp,
        )
