"""
OCCUR-CAM Employee Models
Data models and validation for employee information.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

@dataclass
class EmployeeProfile:
    """Employee profile data model."""
    employee_id: str
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    is_active: bool = True
    hire_date: Optional[datetime] = None
    termination_date: Optional[datetime] = None
    face_photo_path: Optional[str] = None
    face_quality_score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "employee_id": self.employee_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "department": self.department,
            "position": self.position,
            "is_active": self.is_active,
            "hire_date": self.hire_date,
            "termination_date": self.termination_date,
            "face_photo_path": self.face_photo_path,
            "face_quality_score": self.face_quality_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmployeeProfile":
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate employee data and return list of errors."""
        errors = []
        
        if not self.employee_id or len(self.employee_id.strip()) == 0:
            errors.append("Employee ID is required")
        
        if not self.first_name or len(self.first_name.strip()) == 0:
            errors.append("First name is required")
        
        if not self.last_name or len(self.last_name.strip()) == 0:
            errors.append("Last name is required")
        
        if self.email and "@" not in self.email:
            errors.append("Invalid email format")
        
        if self.face_quality_score is not None and (self.face_quality_score < 0 or self.face_quality_score > 1):
            errors.append("Face quality score must be between 0 and 1")
        
        return errors

@dataclass
class FaceEmbedding:
    """Face embedding data model."""
    employee_id: str
    embedding_vector: List[float]
    face_photo_path: str
    quality_score: float
    created_at: datetime
    model_version: str = "buffalo_l"
    
    def to_json(self) -> str:
        """Convert embedding to JSON string for database storage."""
        return json.dumps({
            "embedding": self.embedding_vector,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat()
        })
    
    @classmethod
    def from_json(cls, employee_id: str, json_str: str, face_photo_path: str, quality_score: float) -> "FaceEmbedding":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            employee_id=employee_id,
            embedding_vector=data["embedding"],
            face_photo_path=face_photo_path,
            quality_score=quality_score,
            created_at=datetime.fromisoformat(data["created_at"]),
            model_version=data.get("model_version", "buffalo_l")
        )
    
    def validate(self) -> List[str]:
        """Validate face embedding data."""
        errors = []
        
        if not self.employee_id:
            errors.append("Employee ID is required")
        
        if not self.embedding_vector or len(self.embedding_vector) == 0:
            errors.append("Embedding vector is required")
        
        if not self.face_photo_path:
            errors.append("Face photo path is required")
        
        if self.quality_score < 0 or self.quality_score > 1:
            errors.append("Quality score must be between 0 and 1")
        
        return errors

@dataclass
class EmployeeSiteAssignment:
    """Employee site assignment data model."""
    employee_id: str
    site_id: int
    role: str = "employee"
    access_level: str = "standard"
    assigned_at: datetime = None
    unassigned_at: Optional[datetime] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.assigned_at is None:
            self.assigned_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "employee_id": self.employee_id,
            "site_id": self.site_id,
            "role": self.role,
            "access_level": self.access_level,
            "assigned_at": self.assigned_at,
            "unassigned_at": self.unassigned_at,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmployeeSiteAssignment":
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate assignment data."""
        errors = []
        
        if not self.employee_id:
            errors.append("Employee ID is required")
        
        if not self.site_id:
            errors.append("Site ID is required")
        
        valid_roles = ["employee", "manager", "admin", "security"]
        if self.role not in valid_roles:
            errors.append(f"Role must be one of: {', '.join(valid_roles)}")
        
        valid_access_levels = ["standard", "elevated", "admin"]
        if self.access_level not in valid_access_levels:
            errors.append(f"Access level must be one of: {', '.join(valid_access_levels)}")
        
        return errors

@dataclass
class EmployeeSearchCriteria:
    """Search criteria for employee queries."""
    employee_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None
    has_face_data: Optional[bool] = None
    site_id: Optional[int] = None
    limit: int = 100
    offset: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for query building."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class EmployeeManager:
    """Employee management operations."""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def create_employee(self, profile: EmployeeProfile) -> bool:
        """Create a new employee."""
        try:
            # Validate profile
            errors = profile.validate()
            if errors:
                raise ValueError(f"Validation errors: {', '.join(errors)}")
            
            # Check if employee already exists
            existing = self.db.query(Employee).filter(
                Employee.employee_id == profile.employee_id
            ).first()
            
            if existing:
                raise ValueError(f"Employee with ID {profile.employee_id} already exists")
            
            # Create employee record
            employee_data = profile.to_dict()
            employee = Employee(**employee_data)
            self.db.add(employee)
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def get_employee(self, employee_id: str) -> Optional[EmployeeProfile]:
        """Get employee by ID."""
        try:
            employee = self.db.query(Employee).filter(
                Employee.employee_id == employee_id
            ).first()
            
            if not employee:
                return None
            
            return EmployeeProfile.from_dict(employee.to_dict())
            
        except Exception as e:
            raise e
    
    def search_employees(self, criteria: EmployeeSearchCriteria) -> List[EmployeeProfile]:
        """Search employees based on criteria."""
        try:
            query = self.db.query(Employee)
            
            # Apply filters
            if criteria.employee_id:
                query = query.filter(Employee.employee_id.contains(criteria.employee_id))
            
            if criteria.first_name:
                query = query.filter(Employee.first_name.contains(criteria.first_name))
            
            if criteria.last_name:
                query = query.filter(Employee.last_name.contains(criteria.last_name))
            
            if criteria.email:
                query = query.filter(Employee.email.contains(criteria.email))
            
            if criteria.department:
                query = query.filter(Employee.department == criteria.department)
            
            if criteria.is_active is not None:
                query = query.filter(Employee.is_active == criteria.is_active)
            
            if criteria.has_face_data is not None:
                if criteria.has_face_data:
                    query = query.filter(Employee.face_embedding.isnot(None))
                else:
                    query = query.filter(Employee.face_embedding.is_(None))
            
            # Apply pagination
            query = query.offset(criteria.offset).limit(criteria.limit)
            
            # Execute query
            employees = query.all()
            
            return [EmployeeProfile.from_dict(emp.to_dict()) for emp in employees]
            
        except Exception as e:
            raise e
    
    def update_employee(self, employee_id: str, updates: Dict[str, Any]) -> bool:
        """Update employee information."""
        try:
            employee = self.db.query(Employee).filter(
                Employee.employee_id == employee_id
            ).first()
            
            if not employee:
                raise ValueError(f"Employee {employee_id} not found")
            
            # Update fields
            for key, value in updates.items():
                if hasattr(employee, key):
                    setattr(employee, key, value)
            
            employee.updated_at = datetime.now()
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def deactivate_employee(self, employee_id: str) -> bool:
        """Deactivate an employee."""
        try:
            employee = self.db.query(Employee).filter(
                Employee.employee_id == employee_id
            ).first()
            
            if not employee:
                raise ValueError(f"Employee {employee_id} not found")
            
            employee.is_active = False
            employee.termination_date = datetime.now()
            employee.updated_at = datetime.now()
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise e
