from nexus.orchestration.kernel import OrchestrationKernel

def test_full_50_step_simulation():
    kernel = OrchestrationKernel()
    
    final_state = kernel.run_exploration(
        initial_context="Initialized in undocumented Kubernetes cluster namespace 'legacy-prod'.",
        max_steps=50 # Milestone 2 requirement
    )
    
    assert final_state.step_count == 50
    assert len(final_state.history) == 50
    print("Milestone 2 Simulation Complete!")