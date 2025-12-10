.. _recommender:

#################
Recommenders
#################

.. toctree::
   :hidden:
   :maxdepth: 2

   general
   context
   sequential
   proxy
   guide/index

WarpRec provides a comprehensive collection of **built-in recommender algorithms**, covering both classical and state-of-the-art approaches widely used in the recommendation systems domain.
These models are designed to be **ready-to-use**, serving as a strong baseline for experimentation, benchmarking, and real-world deployment.

The available recommenders are organized into two main categories, reflecting different modeling paradigms:

- **General Recommenders**: These models are versatile and can be applied to a wide range of recommendation tasks. They include algorithms like Matrix Factorization, Neural Collaborative Filtering, and more.
- **Sequential Recommenders**: Designed to capture the temporal dynamics of user behavior, these models focus on the order of interactions. Examples include Recurrent Neural Networks (RNNs) and Transformer-based architectures.

All built-in models are implemented with a focus on **efficiency and scalability**, leveraging **WarpRec’s optimized training backend** to support large-scale datasets and high-performance execution.
Users can flexibly configure architectures and hyperparameters, making it straightforward to adapt each model to specific recommendation scenarios.

Furthermore, WarpRec’s **modular design** makes it easy to extend the framework with **custom recommenders**. By inheriting from the provided base classes and reusing existing components (e.g., interaction handlers, loss modules, evaluation pipelines), researchers and practitioners can rapidly prototype new algorithms while maintaining integration with the training and evaluation ecosystem.

This combination of **ready-to-use models**, **efficient implementations**, and **customization support** ensures that WarpRec can serve both as a practical toolkit for applied recommendation systems and as a flexible research platform for advancing new methodologies.
