"""
wrappers.py — Non-intrusive adapters for communication measurement
===================================================================
These wrappers sit between the simulation runner and the real
agents/aggregators. They delegate ALL real work to the originals
and only observe parameter dictionaries to log byte sizes.

    ┌──────────────┐      ┌───────────────┐
    │ Simulation   │─────▶│ Instrumented  │──▶ real Agent
    │   Runner     │      │    Agent      │     (unchanged)
    └──────────────┘      └───────────────┘
                              │ logs to
                              ▼
                        NetworkSimulator

No existing agent, aggregator, or training code is modified.
"""

from network_sim.network_simulator import (
    NetworkSimulator, AgentNode, EdgeNode, CloudNode,
    Node, measure_params_size,
)


class InstrumentedAgent:
    """
    Wraps any BaseAgent to measure communication when parameters
    are sent (get_parameters) or received (set_parameters).

    All attribute access is transparently delegated to the real agent,
    so this wrapper is a drop-in replacement in the simulation loop.

    How wrapping works:
        - get_parameters(): calls real agent, then logs the upload
          (agent → destination node) via the NetworkSimulator.
        - set_parameters(): logs the download (source → agent),
          then calls the real agent to apply the weights.
        - Every other attribute (get_action, update, etc.) is forwarded
          directly to the real agent via __getattr__.
    """

    def __init__(self, real_agent, agent_node: AgentNode,
                 simulator: NetworkSimulator):
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, '_real_agent', real_agent)
        object.__setattr__(self, '_agent_node', agent_node)
        object.__setattr__(self, '_simulator', simulator)
        object.__setattr__(self, '_upload_target', None)   # set by runner

    def set_upload_target(self, target: Node):
        """Set the node this agent uploads to (edge or cloud)."""
        object.__setattr__(self, '_upload_target', target)

    def get_parameters(self):
        """
        Call the real agent's get_parameters(), then log the upload.
        WHERE COMMUNICATION IS SIMULATED: here we measure the model
        parameter dict size and record the agent→target transfer cost.
        """
        params = self._real_agent.get_parameters()
        # Log upload: agent → edge (or cloud in cloud-only mode)
        if self._upload_target is not None:
            self._simulator.log_transfer(
                src=self._agent_node,
                dst=self._upload_target,
                params=params,
                direction='upload',
            )
        return params

    def set_parameters(self, parameters, log_transfer: bool = True):
        """
        Log the download, then call the real agent's set_parameters().
        WHERE COMMUNICATION IS SIMULATED: here we measure the global
        model being sent back down to the agent.
        """
        # Log download: source → agent
        if log_transfer and self._upload_target is not None:
            self._simulator.log_transfer(
                src=self._upload_target,  # comes from edge/cloud
                dst=self._agent_node,
                params=parameters,
                direction='download',
            )
        self._real_agent.set_parameters(parameters)

    def __getattr__(self, name):
        """Delegate everything else to the real agent (transparent)."""
        return getattr(self._real_agent, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting to the real agent."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._real_agent, name, value)


class InstrumentedEdge:
    """
    Wraps an EdgeAggregator to log edge→cloud communication
    when aggregated parameters are produced.

    The wrapper does NOT modify any data — it only observes.

    How wrapping works:
        - collect(): forwarded directly to the real edge.
        - aggregate(): calls real edge, measures the aggregated
          params, and logs the edge→cloud transfer.
    """

    def __init__(self, real_edge, edge_node: EdgeNode,
                 cloud_node: CloudNode, simulator: NetworkSimulator):
        object.__setattr__(self, '_real_edge', real_edge)
        object.__setattr__(self, '_edge_node', edge_node)
        object.__setattr__(self, '_cloud_node', cloud_node)
        object.__setattr__(self, '_simulator', simulator)

    def collect(self, vehicle_id, params, n_samples):
        """Forward to real edge — agent→edge logging is handled by InstrumentedAgent."""
        self._real_edge.collect(vehicle_id, params, n_samples)

    def aggregate(self):
        """
        Call real edge aggregation, then log the edge→cloud transfer.
        WHERE COMMUNICATION IS SIMULATED: the aggregated model from
        this edge is measured and the transfer cost to cloud is recorded.
        """
        params, n_samples = self._real_edge.aggregate()
        if params is not None:
            self._simulator.log_transfer(
                src=self._edge_node,
                dst=self._cloud_node,
                params=params,
                direction='upload',
            )
        return params, n_samples

    @property
    def vehicle_ids(self):
        return self._real_edge.vehicle_ids

    @property
    def edge_id(self):
        return self._real_edge.edge_id

    def __getattr__(self, name):
        return getattr(self._real_edge, name)


class InstrumentedServer:
    """
    Wraps a FederatedServer to log cloud→edge (or cloud→agent)
    broadcast communication when the global model is distributed.

    How wrapping works:
        - initialize(): forwarded directly.
        - aggregate(): calls real server, then logs the broadcast
          of the global model to all destination nodes.
    """

    def __init__(self, real_server, cloud_node: CloudNode,
                 broadcast_targets: list, simulator: NetworkSimulator):
        """
        Args:
            real_server: The actual FederatedServer instance.
            cloud_node: CloudNode representing the cloud.
            broadcast_targets: List of Nodes that receive the global model
                              (edges in hierarchical, agents in cloud-only).
            simulator: NetworkSimulator for logging.
        """
        object.__setattr__(self, '_real_server', real_server)
        object.__setattr__(self, '_cloud_node', cloud_node)
        object.__setattr__(self, '_broadcast_targets', broadcast_targets)
        object.__setattr__(self, '_simulator', simulator)

    def initialize(self, params):
        """Forward to real server."""
        self._real_server.initialize(params)

    def aggregate(self, edge_updates):
        """
        Call real server aggregation, then log the broadcast.
        WHERE COMMUNICATION IS SIMULATED: after aggregation, the global
        model is broadcast back. We log one download per target node.
        """
        global_params = self._real_server.aggregate(edge_updates)

        # Log broadcast: cloud → each target (edge or agent)
        if global_params is not None:
            for target in self._broadcast_targets:
                self._simulator.log_transfer(
                    src=self._cloud_node,
                    dst=target,
                    params=global_params,
                    direction='download',
                )
        return global_params

    @property
    def global_params(self):
        return self._real_server.global_params

    def __getattr__(self, name):
        return getattr(self._real_server, name)
