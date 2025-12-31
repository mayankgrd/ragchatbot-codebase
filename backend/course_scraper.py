"""
DeepLearning.AI Course Scraper

Scrapes course content from learn.deeplearning.ai to generate course scripts
compatible with the RAG system's document processor.
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup


class DeepLearningAIScraper:
    """Scraper for DeepLearning.AI course content"""

    BASE_URL = "https://learn.deeplearning.ai"

    def __init__(self, session_cookie: str):
        """
        Initialize scraper with authentication.

        Args:
            session_cookie: User's session cookie value from browser
        """
        self.session = requests.Session()

        # Set up cookies - try common cookie names
        self.session.cookies.set('session', session_cookie, domain='learn.deeplearning.ai')
        self.session.cookies.set('__session', session_cookie, domain='learn.deeplearning.ai')

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://learn.deeplearning.ai/',
        })

    def scrape_course(self, course_url: str) -> Tuple[str, Optional[str]]:
        """
        Scrape a full course and return script content.

        Args:
            course_url: URL like https://learn.deeplearning.ai/courses/course-name/lesson/xyz/intro

        Returns:
            Tuple of (script_content, error_message)
            If successful, error_message is None
            If failed, script_content is empty string
        """
        try:
            # Extract course slug from URL
            course_slug = self._extract_course_slug(course_url)
            if not course_slug:
                return "", "Could not extract course identifier from URL"

            # Fetch the course page
            response = self.session.get(course_url, timeout=30)
            if response.status_code == 401 or response.status_code == 403:
                return "", "Authentication failed. Please check your session cookie."
            if response.status_code != 200:
                return "", f"Failed to fetch course page: HTTP {response.status_code}"

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract course metadata
            metadata = self._extract_course_metadata(soup, course_url)
            if not metadata.get('title'):
                return "", "Could not extract course title from page"

            # Get lesson list
            lessons = self._get_lesson_list(soup, course_url)
            if not lessons:
                return "", "Could not find any lessons in the course"

            # Scrape each lesson's transcript
            scraped_lessons = []
            for i, lesson in enumerate(lessons):
                print(f"Scraping lesson {i}: {lesson.get('title', 'Unknown')}")

                transcript = self._scrape_lesson_transcript(lesson['url'])
                if transcript:
                    scraped_lessons.append({
                        'number': i,
                        'title': lesson.get('title', f'Lesson {i}'),
                        'url': lesson['url'],
                        'transcript': transcript
                    })

                # Be polite - don't hammer the server
                time.sleep(1)

            if not scraped_lessons:
                return "", "Could not scrape any lesson transcripts"

            # Format as course script
            script_content = self._format_as_script(metadata, scraped_lessons)
            return script_content, None

        except requests.exceptions.Timeout:
            return "", "Request timed out. The server may be slow or unavailable."
        except requests.exceptions.ConnectionError:
            return "", "Could not connect to the server. Please check your internet connection."
        except Exception as e:
            return "", f"Scraping error: {str(e)}"

    def _extract_course_slug(self, url: str) -> Optional[str]:
        """Extract course slug from URL"""
        # URL format: https://learn.deeplearning.ai/courses/course-name/lesson/xyz/...
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        if len(path_parts) >= 2 and path_parts[0] == 'courses':
            return path_parts[1]
        return None

    def _extract_course_metadata(self, soup: BeautifulSoup, course_url: str) -> Dict:
        """Extract course title, instructor, and link from page"""
        metadata = {
            'title': '',
            'instructor': '',
            'course_link': course_url
        }

        # Try various selectors for title
        title_selectors = [
            'h1',
            '[class*="course-title"]',
            '[class*="courseTitle"]',
            '[data-testid="course-title"]',
            'meta[property="og:title"]',
            'title'
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                if selector.startswith('meta'):
                    text = element.get('content', '')
                else:
                    text = element.get_text(strip=True)

                if text and len(text) > 3:
                    # Clean up title
                    text = re.sub(r'\s*\|.*$', '', text)  # Remove "| DeepLearning.AI"
                    text = re.sub(r'\s*-\s*DeepLearning\.AI.*$', '', text, flags=re.IGNORECASE)
                    metadata['title'] = text.strip()
                    break

        # Try to find instructor
        instructor_selectors = [
            '[class*="instructor"]',
            '[class*="author"]',
            '[class*="teacher"]',
            'a[href*="/instructors/"]',
        ]

        for selector in instructor_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text and len(text) > 2 and len(text) < 100:
                    metadata['instructor'] = text
                    break

        # Extract base course link (remove lesson-specific parts)
        parsed = urlparse(course_url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'courses':
            base_path = f"/courses/{path_parts[1]}"
            metadata['course_link'] = f"{parsed.scheme}://{parsed.netloc}{base_path}"

        return metadata

    def _get_lesson_list(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract list of lessons from course page"""
        lessons = []
        parsed_base = urlparse(base_url)

        # Try various selectors for lesson navigation
        nav_selectors = [
            'nav a[href*="/lesson/"]',
            '[class*="sidebar"] a[href*="/lesson/"]',
            '[class*="navigation"] a[href*="/lesson/"]',
            '[class*="lesson-list"] a',
            '[class*="lessonList"] a',
            'a[href*="/lesson/"]',
        ]

        seen_urls = set()

        for selector in nav_selectors:
            elements = soup.select(selector)
            for elem in elements:
                href = elem.get('href', '')
                if not href or href in seen_urls:
                    continue

                # Make absolute URL
                if href.startswith('/'):
                    full_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                elif not href.startswith('http'):
                    full_url = urljoin(base_url, href)
                else:
                    full_url = href

                # Only include lesson URLs
                if '/lesson/' not in full_url:
                    continue

                seen_urls.add(full_url)

                # Get lesson title
                title = elem.get_text(strip=True)
                if not title:
                    title = f"Lesson {len(lessons)}"

                lessons.append({
                    'url': full_url,
                    'title': title
                })

            if lessons:
                break

        return lessons

    def _scrape_lesson_transcript(self, lesson_url: str) -> Optional[str]:
        """Scrape transcript from a lesson page"""
        try:
            response = self.session.get(lesson_url, timeout=30)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Try various selectors for transcript content
            transcript_selectors = [
                '[class*="transcript"]',
                '[class*="Transcript"]',
                '[data-testid="transcript"]',
                '[class*="video-transcript"]',
                '[class*="lesson-content"]',
                '[class*="lessonContent"]',
                'article',
                '.content',
                'main',
            ]

            for selector in transcript_selectors:
                element = soup.select_one(selector)
                if element:
                    # Get text content
                    text = element.get_text(separator=' ', strip=True)

                    # Clean up the text
                    text = self._clean_transcript(text)

                    if text and len(text) > 100:  # Minimum viable transcript
                        return text

            # Fallback: try to find any substantial text content
            paragraphs = soup.find_all('p')
            if paragraphs:
                texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50]
                if texts:
                    return self._clean_transcript(' '.join(texts))

            return None

        except Exception as e:
            print(f"Error scraping lesson {lesson_url}: {e}")
            return None

    def _clean_transcript(self, text: str) -> str:
        """Clean up transcript text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common UI elements
        ui_patterns = [
            r'Show Transcript',
            r'Hide Transcript',
            r'Play Video',
            r'Pause Video',
            r'\d+:\d+',  # Timestamps
            r'Next Lesson',
            r'Previous Lesson',
            r'Mark as Complete',
        ]

        for pattern in ui_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up whitespace again
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _format_as_script(self, metadata: Dict, lessons: List[Dict]) -> str:
        """Format scraped data into course script format"""
        lines = []

        # Course metadata
        lines.append(f"Course Title: {metadata['title']}")
        lines.append(f"Course Link: {metadata['course_link']}")
        lines.append(f"Course Instructor: {metadata.get('instructor', 'Unknown')}")
        lines.append("")

        # Lessons
        for lesson in lessons:
            lines.append(f"Lesson {lesson['number']}: {lesson['title']}")
            lines.append(f"Lesson Link: {lesson['url']}")
            lines.append(lesson['transcript'])
            lines.append("")

        return '\n'.join(lines)
