from math import prod
import cvxpy as cp
import numpy as np
from copy import deepcopy
from toqito.channels import partial_trace
from toqito.matrix_ops import tensor
from toqito.perms import permute_systems, permutation_operator
from toqito.rand import random_density_matrix

from cvxpy import Variable
from typing import Any, Optional, Dict
from dataclasses import dataclass, field
   

### This code is based on the qsuperop matlab library of Alastair Abbott.
### https://github.com/alastair-abbott/qsuperops

def _group_future(W, dims, parties):
    """
    Group all future spaces into a single space for QCFO projection.
    
    In QCFO theory, future spaces (typically the final output systems) need to be
    treated as a single composite system for applying trace-replace conditions.
    This function reshapes the space structure accordingly.
    
    Args:
        W: Quantum superoperator matrix (numpy array or cvxpy Variable)
        dims: List of dimensions for each quantum system
        parties: List of lists defining which systems belong to each party
        
    Returns:
        tuple: (W, new_dims, new_parties) where future spaces are grouped
               If only one future space exists, returns input unchanged
    """
    # Check if future space is already single
    size_future = len(parties[-1])
    if size_future == 1:
        return W, dims, parties
    
    else:
        # Combine all future space dimensions into one
        dim_future = np.prod(dims[-size_future:])
        new_dims = dims[:-size_future] + [dim_future]
        
        # Update party structure to reflect grouped future space
        new_parties = deepcopy(parties)
        new_parties[-1] = len(new_dims)
        
        return W, new_dims, new_parties
        
def project_onto_QCFOs(W, dims_raw, parties_raw):
    """
    Project a quantum superoperator onto the space of QCFOs.
    
    QCFOs (Quantum Circuit with Fixed Order) satisfy specific causality constraints
    enforced through trace-replace conditions. This projection ensures the
    superoperator respects the fixed causal ordering between systems.
    
    The projection applies trace-replace operations iteratively, working backwards
    through the causal order to enforce that intermediate systems cannot influence
    their own past.
    
    Args:
        W: Input quantum superoperator matrix
        dims_raw: List of dimensions for each quantum system  
        parties_raw: List of lists defining the party structure and causal ordering
        
    Returns:
        numpy.ndarray or cvxpy.Variable: Projected superoperator satisfying QCFO constraints
        
    Note:
        Assumes canonical ordering where future space is identified as single space
    """
    # Number of intermediate systems in causal chain
    N = len(parties_raw) - 2
    F = 2*N + 2
    
    # The future spaces are grouped together, as a single space, for the trace and replace conditions.
    W, dims, parties = _group_future(W, dims_raw, parties_raw)           
        
    W_proj = W
    
    # Apply trace-replace conditions working backwards through causal order
    for n in range(N, -1, -1):
        # Define system indices for trace-replace operation
        sys1 = np.array(range(2*n+1, F))  # Systems from current level forward
        sys2 = np.array(range(2*n, F))    # Systems from previous level forward
                
        # Filter out trivial (dimension 1) systems
        dims1_filtered = [dims[i] for i in sys1 if dims[i] != 1]
        dims2_filtered = [dims[i] for i in sys2 if dims[i] != 1]

        # Apply trace-replace only if system structures differ
        # (Tracing out 1 dimensional systems doesn't do anything)
        if dims1_filtered != dims2_filtered:
            W_proj = W_proj - (tr_replace(W_proj, list(sys1), dims) - tr_replace(W_proj, list(sys2), dims))

    return W_proj

def tr_replace(W, sys, dims):
    """
    Trace and replace operation for QCFO constraints.
    
    This operation traces out specified systems and replaces them with maximally
    mixed states, implementing the trace-replace conditions that enforce causal
    structure in QCFOs.
    
    Mathematically: Tr_sys(W) ⊗ (I/d_sys) where Tr_sys is partial trace over sys
    and I/d_sys is the maximally mixed state on the traced systems.
    
    Args:
        W: Input quantum superoperator (numpy array or cvxpy Variable)
        sys: List of system indices to trace out
        dims: List of dimensions for all systems
        
    Returns:
        numpy.ndarray or cvxpy.Variable: Result of trace-replace operation
        
    Note:
        Handles edge cases of empty systems, trivial dimensions, and full traces
    """
    d = np.prod(dims)

    # If no systems to trace out, return the original superoperator
    if not sys:
        return W
    
    # Compute dimension of systems being traced out
    d_traced = np.prod([dims[s] for s in sys])
    
    # If all systems are dim 1 or sys is empty, do nothing
    if d_traced == 1:
        return W
    
    # If tracing out all systems, return scaled identity
    if d_traced == d:
        if isinstance(W, np.ndarray):
            return np.trace(W) * np.eye(d) / d        
        else:
            return cp.trace(W) * np.eye(d) / d
    
    # Perform partial trace and tensor with maximally mixed state
    if isinstance(W, np.ndarray):
        W_traced = partial_trace(W, sys, dims)
        id_traced = np.eye(d_traced) / d_traced  # Maximally mixed state
        result = np.kron(W_traced, id_traced)        
    else:
        # Handle cvxpy Variables
        W_traced = partial_trace_multiple(W, dims, sys)
        id_traced = np.eye(d_traced) / d_traced
        result = cp.kron(W_traced, id_traced)  
    
    # Get remaining systems (not traced out)
    rest = [s for s in range(len(dims)) if s not in sys]
    
    # Permute back to original system ordering
    perm = rest + sys  # Put remaining systems first, traced systems last
    dims_reordered = [dims[s] for s in perm]

    # Apply permutation to restore original ordering
    p_matrix = permutation_operator(dims_reordered, perm, True, False)
    result = p_matrix @ result @ p_matrix.T

    return result

def pure_CJ(A):
    """
    Pure Choi-Jamiołkowski isomorphism for quantum channels.
    
    Converts a quantum channel matrix A into its Choi representation using
    the standard maximally entangled state |Φ⁺⟩ = Σᵢ |i⟩⊗|i⟩.
    
    The Choi matrix is constructed as (I ⊗ A)|Φ⁺⟩⟨Φ⁺|, which for pure states
    reduces to (I ⊗ A)|Φ⁺⟩.
    
    Args:
        A: Input quantum channel matrix of shape (d, d)
        
    Returns:
        numpy.ndarray: Choi vector representation of shape (d², 1)
        
    Note:
        Returns vectorized form rather than matrix
    """
    d = np.shape(A)[0]
    
    # Construct maximally entangled state |Φ⁺⟩ = Σᵢ |i⟩⊗|i⟩
    id_CJ = np.zeros((d**2,1))
    for i in range(d):
        # Create computational basis vector |i⟩
        basis_vec = np.zeros((d, 1))
        basis_vec[i] = 1
        # Add |i⟩⊗|i⟩ to superposition
        id_CJ += np.kron(basis_vec, basis_vec)

    # Apply (I ⊗ A) to maximally entangled state
    A_CJ = np.kron(np.eye(d), A) @ id_CJ
    return A_CJ

def measure_output(Wr, dims, parties, M, F_sys):
    """
    Apply a POVM measurement to output systems of a quantum superinstrument.
    
    Takes a superinstrument (list of superoperators) and applies a Positive
    Operator-Valued Measure (POVM) to specified systems, generating a new
    superinstrument with additional measurement outcomes.
    
    This is used to implement measurement protocols in quantum networks where
    parties perform measurements on their output systems.
    
    Args:
        Wr: Input superinstrument - list of superoperator matrices, or single matrix
        dims: List of dimensions for all quantum systems
        parties: List of lists defining party structure  
        M: List of POVM elements, where each M[r] is a positive semidefinite matrix
           and sum(M) = Identity
        F_sys: List of system indices where measurement is applied
        
    Returns:
        tuple: (Wr_new, dims_news, parties_news) where:
            - Wr_new: New superinstrument as list of matrices
            - dims_news: Updated dimensions with measured systems removed
            - parties_news: Updated party structure with measured systems removed
            
    Note:
        Validates that M forms a proper POVM (positive elements summing to identity)
    """
    # Convert single matrix to list format
    if not isinstance(Wr, list):
        Wr = [Wr]
        r_init = 1
    else:
        r_init = len(Wr)
        
    d = prod(dims)
    d_F_sys = prod([dims[s] for s in F_sys])  # Dimension of measured systems
    R = len(M)  # Number of POVM elements (measurement outcomes)

    # Validate POVM properties
    M_total = np.zeros(np.shape(M[0]), dtype=complex)
    for r in range(R):
        M_total = M_total + M[r]
        # Check each element is positive semidefinite
        assert(np.min(np.linalg.eigvals(M[r])) >= -1e-6)
    
    # Check POVM elements sum to identity
    assert(np.allclose(M_total, np.eye(d_F_sys), 1e-6))

    # Initialize output superinstrument
    # Each input element creates R output elements (one per measurement outcome)
    Wr_new = np.zeros((int(r_init*R), d//d_F_sys, d//d_F_sys))

    # Build measurement operators I ⊗ M_r for each POVM element
    for r in range(R):
        # Create permutation to move measured systems to the end
        perm = list(range(len(dims)))
        for sys in F_sys:
            perm.remove(sys)
        perm = perm + F_sys  # Unmeasured systems first, measured systems last
        
        dims_perm = [dims[s] for s in perm]
        
        # Construct I ⊗ M_r^T and permute to correct position
        # (Transpose needed for correct application in Choi representation)
        idMr = permute_systems(tensor(np.eye(d//d_F_sys), M[r].T), perm, dims_perm, False, True)
        
        # Apply measurement to each input superoperator element
        for i in range(r_init):
            # Apply measurement and trace out measured systems
            Wr_new[i*R + r,:,:] = partial_trace(Wr[i] @ idMr, F_sys, dims)

    # Update dimensions and party structure (remove measured systems)
    dims_news = [dims[s] for s in range(len(dims)) if s not in F_sys]                    
    parties_news = [[s for s in p if s not in F_sys] for p in parties]

    # Convert back to list of matrices
    Wr_new = [Wr_new[i,:,:] for i in range(np.shape(Wr_new)[0])]

    return Wr_new, dims_news, parties_news

def partial_trace_multiple(W: cp.Variable, dims, subsystems):
    """
    Apply partial trace over multiple subsystems for cvxpy Variables.
    
    Sequentially traces out multiple quantum subsystems from a cvxpy Variable
    representing a quantum state or superoperator. Systems are traced in reverse
    order to maintain valid indices throughout the process.
    
    Args:
        W: cvxpy Variable representing quantum operator
        dims: List of dimensions for all subsystems
        subsystems: List of subsystem indices to trace out
        
    Returns:
        cp.Variable: Partially traced cvxpy Variable
        
    Note:
        Modifies dims_temp internally but does not affect input dims list
        Systems traced in reverse order to preserve indexing validity
    """
    W_traced = W
    dims_temp = deepcopy(dims)

    # Trace out subsystems in reverse order so indices stay valid
    for s in sorted(subsystems, reverse=True):
        if dims_temp[s] == 1:
            del dims_temp[s]
            continue  # Skip trivial dimensions

        W_traced = cp.partial_trace(W_traced, dims_temp, s)
        # Remove traced system from dimension list
        del dims_temp[s]

    return W_traced