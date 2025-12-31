from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])

# Will be set by app.py after RAG system is initialized
rag_system = None


def set_rag_system(system):
    """Set the RAG system reference for admin routes"""
    global rag_system
    rag_system = system


class ScrapeRequest(BaseModel):
    """Request model for scraping a course from URL"""
    url: str
    session_cookie: str


class UploadRequest(BaseModel):
    """Request model for uploading course content"""
    content: str


class CourseResponse(BaseModel):
    """Response model for course operations"""
    status: str
    course_title: Optional[str] = None
    lesson_count: Optional[int] = None
    chunk_count: Optional[int] = None
    message: str


class CourseListResponse(BaseModel):
    """Response model for listing all courses"""
    courses: List[Dict[str, Any]]
    total: int


@admin_router.get("/courses", response_model=CourseListResponse)
async def get_all_courses():
    """Get detailed list of all courses with metadata"""
    try:
        courses = rag_system.get_detailed_course_list()
        return CourseListResponse(courses=courses, total=len(courses))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/courses/scrape", response_model=CourseResponse)
async def scrape_course(request: ScrapeRequest):
    """Scrape a course from DeepLearning.AI URL"""
    try:
        from course_scraper import DeepLearningAIScraper

        scraper = DeepLearningAIScraper(request.session_cookie)
        script_content, error = scraper.scrape_course(request.url)

        if error:
            return CourseResponse(status="error", message=error)

        # Add to system
        course, chunk_count, error = rag_system.add_course_from_text(script_content)

        if error:
            return CourseResponse(status="error", message=error)

        return CourseResponse(
            status="success",
            course_title=course.title,
            lesson_count=len(course.lessons),
            chunk_count=chunk_count,
            message="Successfully scraped and indexed course"
        )
    except ImportError:
        return CourseResponse(status="error", message="Course scraper not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/courses/upload", response_model=CourseResponse)
async def upload_course(request: UploadRequest):
    """Upload course content as text"""
    try:
        course, chunk_count, error = rag_system.add_course_from_text(request.content)

        if error:
            return CourseResponse(status="error", message=error)

        return CourseResponse(
            status="success",
            course_title=course.title,
            lesson_count=len(course.lessons),
            chunk_count=chunk_count,
            message="Successfully indexed course"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.delete("/courses/{course_title:path}", response_model=CourseResponse)
async def delete_course(course_title: str):
    """Delete a course from the system"""
    try:
        success, error = rag_system.delete_course(course_title)

        if success:
            return CourseResponse(status="success", message=f"Deleted course '{course_title}'")
        else:
            return CourseResponse(status="error", message=error)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
