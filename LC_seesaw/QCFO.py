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

from QCFO_utils import project_onto_QCFOs, measure_output

@dataclass
class QCFO:
    """
    Quantum Circuit with Fixed Order (QCFO) object.
    
    Represents a quantum superoperator with fixed causal ordering between 
    input/output spaces. Can be numeric (numpy arrays) or symbolic (cvxpy Variables).
    
    Attributes:
        is_numeric: Whether elements contain numeric values or cvxpy Variables
        nb_elements: Number of measurement outcomes/elements in the instrument
        spaces: List defining the space structure for input/output systems
        dimensions: Dimensions of each quantum system
        elements: Dictionary mapping element index to matrix (numeric or symbolic)
        is_real: Whether to use real-valued matrices (for optimization efficiency)
        constraints: List of cvxpy constraints (for symbolic case)
        norm: Computed normalization factor
    """
    is_numeric : bool
    nb_elements: int
    spaces: list[list[int]]
    dimensions: list[int]
    elements : Optional[Dict[int, Any]] = None
    is_real: bool = field(default=False)
    constraints: list[Any] = field(default_factory=list)    
    norm: Optional[int] = field(init=False)
    is_diag: Optional[bool] = field(default=False)

    def __post_init__(self):
        """
        Initialize QCFO after dataclass creation.
        
        Computes norm, validates inputs, and initializes elements either
        randomly (numeric) or as optimization variables (symbolic).
        """
        self.norm = self._compute_norm()

        if self.elements is None and self.is_numeric:
            self.elements = self._random_QCFO()
        elif self.elements is None and not self.is_numeric:
            self.elements = self._init_variables()
        elif self.elements is not None and not self.is_numeric:
            raise TypeError("Cannot receive pre-initialised cvxpy variables.") #To-Do : warm-start
        elif self.elements is not None:
            self._validate_elements()      

    def _compute_norm(self) -> int:
        """
        Compute the normalization factor for the QCFO.
        
        For a QCFO with N intermediate systems, the norm is the product of
        dimensions of the past (input) space and all intermediate output spaces.
        
        Returns:
            Normalization factor as integer
        """
        N = len(self.spaces) - 2

        norm = self.dimensions[self.spaces[0][0]]
        
        for n in range(1,N+1):
            norm *= self.dimensions[self.spaces[n][1]]

        return norm
    
    def _random_QCFO(self) -> Dict[int, np.ndarray]:
        """
        Generate a random valid QCFO instrument.
        
        Creates a random density matrix, projects it onto the QCFO space,
        ensures positive semidefiniteness, and measures to create instrument elements.
        
        Returns:
            Dictionary mapping element indices to numpy arrays
        """

        N = len(self.spaces) - 2

        # First we add an extra space to F that will split the superoperator into an R-element instrument
        dims_extended = self.dimensions + [self.nb_elements]
        total_dimension = np.prod(dims_extended)
        spaces_extended = deepcopy(self.spaces)
        spaces_extended[N+1] = [self.spaces[N+1][0], len(dims_extended)-1]

        dO = self.norm
        d_I = total_dimension//dO

        # Generate random density matrix scaled by output dimension
        W = dO*random_density_matrix(total_dimension, is_real=self.is_real)

        # Project onto QCFO constraint space
        W = project_onto_QCFOs(W,dims_extended,spaces_extended)

        # Make sure it is SDP by mixing with white noise if needed
        eig_min = min(np.linalg.eigvals(W))
        q = eig_min*d_I/(eig_min*d_I-1)
        if eig_min < 0:
            noisyW = np.eye(total_dimension)/d_I
            W = q*noisyW + (1-q)*W

        # Measure to create the number of elements
        if self.nb_elements > 1:
            # Create computational basis measurement
            basis  = np.eye(self.nb_elements)
            M = []
            for r in range(self.nb_elements):
                M.append(np.outer(basis[:,r], basis[:,r]))
            Wr, _, _ = measure_output(W, dims_extended, spaces_extended, M, [len(dims_extended)-1])
        else:
            Wr = [W]

        # Convert to dictionary format
        elements = {}
        for el, W_el in enumerate(Wr):
            elements[el] = W_el
        return elements
        
    def _init_variables(self) -> dict[int, Variable]:
        """
        Initialize symbolic cvxpy Variables for optimization.
        
        Creates optimization variables for each element with appropriate constraints:
        - Positive semidefiniteness 
        - Correct trace normalization
        - QCFO space constraints
        
        Returns:
            Dictionary mapping element indices to cvxpy Variables
            
        Side Effects:
            Adds constraints to self.constraints list
        """
        elements = {}
        d = np.prod(self.dimensions)
        
        # Create variable for each measurement element
        for el in range(self.nb_elements):
            if self.is_real:
                # Real PSD matrices for faster optimization                
                if self.is_diag:
                    diag_entries = cp.Variable(d, nonneg=True)
                    elements[el] = cp.diag(diag_entries)
                else:             
                    elements[el] = cp.Variable((d,d), PSD=True)
            else:
                # Complex hermitian matrices with explicit PSD constraint
                elements[el] = cp.Variable((d,d), hermitian=True)
                self.constraints += [elements[el] >> 0]

        # Sum of all elements (total superoperator)
        W = cp.sum([elements[el] for el in range(self.nb_elements)])
        
        # Normalization constraint
        self.constraints += [cp.trace(W) == cp.Constant(self.norm)]
        
        # QCFO space constraint
        self.constraints += [W == project_onto_QCFOs(W, self.dimensions, self.spaces)]
        
        return elements        

    def _validate_elements(self, tol: float = 1e-6):
        """
        Validate that elements form a proper QCFO instrument.
        
        Checks:
        - Correct number of elements
        - Proper trace normalization
        - QCFO space constraints satisfied
        - Positive semidefiniteness of each element
        
        Args:
            tol: Numerical tolerance for validation checks
            
        Raises:
            AssertionError: If any validation check fails
        """
        W = sum(self.elements.values())
        
        # Check element count
        assert self.nb_elements == len(self.elements)
        
        # Check trace normalization
        assert np.isclose(self.norm, np.trace(W))
        
        # Check QCFO space constraint
        assert np.allclose(project_onto_QCFOs(W, self.dimensions, self.spaces), W, atol=tol)

        # Check each element is PSD and numeric
        for el in range(self.nb_elements):
            assert np.all(np.linalg.eigvals(self.elements[el]) >= -tol)
            assert isinstance(self.elements[el], np.ndarray)

    def to_numeric(self) -> 'QCFO':
        """
        Transform a symbolic QCFO into a numeric one.
        Extracts values from cvxpy Variables after optimization.
    
        Returns:
            A new numeric QCFO with the same structure but numpy arrays as elements
        
        Raises:
            AssertionError: If QCFO is already numeric
            ValueError: If cvxpy Variables have not been optimized (values are None)
        """      
        assert not self.is_numeric, "QCFO is already numeric"
    
        # Extract numeric values from cvxpy Variables
        numeric_elements = {}
        for el in range(self.nb_elements):
            if self.elements[el].value is None:
                raise ValueError(f"Element {el} has not been optimized - no value available")
            numeric_elements[el] = np.array(self.elements[el].value)
    
        # Create new numeric QCFO with extracted values
        return QCFO(
            is_numeric=True,
            nb_elements=self.nb_elements,
            spaces=deepcopy(self.spaces),
            dimensions=deepcopy(self.dimensions), 
            elements=numeric_elements,
            is_real=self.is_real
        )