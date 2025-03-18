from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from goose_mcp import MCPServer, register_function
import json
import os
from datetime import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Models
class ProcessRequest(BaseModel):
    document_ids: List[str]
    batch_size: Optional[int] = None
    priority: Optional[str] = "normal"

class ValidationRequest(BaseModel):
    sample_size: int = 100
    validation_type: str = "random"
    category: Optional[str] = None

class CustomReportRequest(BaseModel):
    format: str = "markdown"
    filters: Optional[Dict[str, Any]] = None
    grouping: Optional[List[str]] = None
    include_details: bool = True

class Config(BaseModel):
    rubric_path: str
    confidence_thresholds: Dict[str, int]
    batch_size: int
    processing: Dict[str, Any]
    llm: Dict[str, Any]

class GDriveClassifierMCP(MCPServer):
    def __init__(self):
        super().__init__()
        self.config = self._load_config()
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.credentials = self._load_credentials()
        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.docs_service = build('docs', 'v1', credentials=self.credentials)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.processing_status = {}

    def _load_config(self) -> Config:
        config_path = os.getenv('CONFIG_PATH', 'config.json')
        with open(config_path, 'r') as f:
            return Config(**json.load(f))

    def _load_credentials(self) -> Credentials:
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Google OAuth token not found at {token_path}")
        
        with open(token_path, 'r') as token:
            token_data = json.load(token)
            return Credentials.from_authorized_user_info(token_data)

    @register_function(
        "Process documents for classification with progress updates",
        request_model=ProcessRequest,
        streaming=True
    )
    async def process_documents(self, request: ProcessRequest):
        """Process documents with real-time progress updates."""
        batch_size = request.batch_size or self.config.batch_size
        total_docs = len(request.document_ids)
        processed = 0
        
        # Initialize processing status
        task_id = datetime.now().isoformat()
        self.processing_status[task_id] = {
            'total': total_docs,
            'processed': 0,
            'successful': 0,
            'failed': 0
        }
        
        try:
            # Process documents in batches
            for i in range(0, total_docs, batch_size):
                batch = request.document_ids[i:i + batch_size]
                
                # Process batch concurrently
                tasks = []
                for doc_id in batch:
                    task = asyncio.create_task(self._process_document(doc_id))
                    tasks.append(task)
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update status and yield progress
                processed += len(batch)
                successful = sum(1 for r in batch_results if not isinstance(r, Exception))
                failed = len(batch) - successful
                
                self.processing_status[task_id].update({
                    'processed': processed,
                    'successful': successful,
                    'failed': failed
                })
                
                yield {
                    'task_id': task_id,
                    'progress': {
                        'processed': processed,
                        'total': total_docs,
                        'percentage': (processed / total_docs) * 100,
                        'successful': successful,
                        'failed': failed
                    },
                    'batch_results': [
                        r if not isinstance(r, Exception) else {'error': str(r)}
                        for r in batch_results
                    ]
                }
        
        except Exception as e:
            yield {
                'task_id': task_id,
                'error': str(e),
                'progress': self.processing_status[task_id]
            }
        
        finally:
            # Cleanup status after completion
            if task_id in self.processing_status:
                del self.processing_status[task_id]

    async def _process_document(self, doc_id: str) -> Dict:
        """Process a single document."""
        try:
            # Get document content
            content = await self._get_document_content(doc_id)
            
            # Get document metadata
            metadata = await self._get_document_metadata(doc_id)
            
            # Classify content
            classification = await self._classify_content(content, metadata)
            
            # Update document properties
            await self._update_document_properties(doc_id, classification)
            
            return {
                'document_id': doc_id,
                'classification': classification,
                'processed_at': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            return {
                'document_id': doc_id,
                'error': str(e),
                'processed_at': datetime.now().isoformat(),
                'success': False
            }

    async def _get_document_content(self, doc_id: str) -> str:
        """Get document content from Google Drive."""
        loop = asyncio.get_event_loop()
        
        # Get file metadata
        file = await loop.run_in_executor(
            self.executor,
            lambda: self.drive_service.files().get(fileId=doc_id, fields='mimeType').execute()
        )
        
        if file['mimeType'] == 'application/vnd.google-apps.document':
            # Get Google Doc content
            doc = await loop.run_in_executor(
                self.executor,
                lambda: self.docs_service.documents().get(documentId=doc_id).execute()
            )
            
            content = []
            for elem in doc.get('body', {}).get('content', []):
                if 'paragraph' in elem:
                    for para_elem in elem['paragraph']['elements']:
                        if 'textRun' in para_elem:
                            content.append(para_elem['textRun']['content'])
            
            return '\n'.join(content)
        else:
            # Get regular file content
            request = self.drive_service.files().get_media(fileId=doc_id)
            content = await loop.run_in_executor(self.executor, request.execute)
            return content.decode('utf-8')

    async def _get_document_metadata(self, doc_id: str) -> Dict:
        """Get document metadata from Google Drive."""
        loop = asyncio.get_event_loop()
        file = await loop.run_in_executor(
            self.executor,
            lambda: self.drive_service.files().get(
                fileId=doc_id,
                fields='id, name, mimeType, createdTime, modifiedTime, owners'
            ).execute()
        )
        return file

    async def _classify_content(self, content: str, metadata: Dict) -> Dict:
        """Classify document content using LLM."""
        # Load classification rubric
        with open(self.config.rubric_path, 'r') as f:
            rubric = json.load(f)
        
        # Prepare the prompt
        prompt = self._build_classification_prompt(content, metadata, rubric)
        
        # Get classification from LLM
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.openai_client.chat.completions.create(
                model=self.config.llm['model'],
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.llm['temperature'],
                max_tokens=self.config.llm['max_tokens']
            )
        )
        
        # Parse and enhance classification
        classification = self._parse_classification_response(
            response.choices[0].message.content
        )
        classification = self._add_confidence_levels(classification)
        
        return classification

    def _build_classification_prompt(self, content: str, metadata: Dict, rubric: Dict) -> str:
        """Build the prompt for document classification."""
        prompt_parts = [
            "Please classify the following document according to the provided rubric.",
            "\nDocument content:",
            content[:4000]  # Limit content length
        ]
        
        if metadata:
            prompt_parts.extend([
                "\nDocument metadata:",
                json.dumps(metadata, indent=2)
            ])
            
        prompt_parts.extend([
            "\nClassification rubric:",
            json.dumps(rubric, indent=2),
            "\nPlease provide the classification in JSON format with the following structure:",
            """{
                "categories": [
                    {
                        "name": "category_name",
                        "confidence": 0-100,
                        "reasoning": "explanation"
                    }
                ],
                "overall_confidence": 0-100,
                "summary": "brief classification summary"
            }"""
        ])
        
        return "\n".join(prompt_parts)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are a document classification expert. Your task is to:
1. Analyze document content and metadata
2. Match it against the provided classification rubric
3. Return structured classification results
4. Provide confidence scores and reasoning
Be precise and follow the rubric exactly."""

    def _parse_classification_response(self, response: str) -> Dict:
        """Parse the LLM's response into a structured format."""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            # Parse the JSON
            classification = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['categories', 'overall_confidence', 'summary']
            for field in required_fields:
                if field not in classification:
                    raise ValueError(f"Missing required field: {field}")
            
            return classification
            
        except Exception as e:
            return {
                "categories": [],
                "overall_confidence": 0,
                "summary": "Error parsing classification",
                "error": str(e)
            }

    def _add_confidence_levels(self, classification: Dict) -> Dict:
        """Add confidence level labels based on configured thresholds."""
        thresholds = self.config.confidence_thresholds
        
        def get_confidence_level(score: int) -> str:
            if score >= thresholds['high']:
                return "HIGH"
            elif score >= thresholds['medium']:
                return "MEDIUM"
            else:
                return "LOW"
        
        # Add confidence levels to categories
        for category in classification['categories']:
            category['confidence_level'] = get_confidence_level(category['confidence'])
        
        # Add overall confidence level
        classification['overall_confidence_level'] = get_confidence_level(
            classification['overall_confidence']
        )
        
        return classification

    async def _update_document_properties(self, doc_id: str, classification: Dict):
        """Update document properties with classification results."""
        properties = {
            'classified': 'true',
            'classification_date': datetime.now().isoformat(),
            'classification_summary': classification['summary'],
            'overall_confidence': str(classification['overall_confidence']),
            'categories': ','.join([c['name'] for c in classification['categories']])
        }
        
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.drive_service.files().update(
                fileId=doc_id,
                body={'properties': properties}
            ).execute()
        )

    @register_function(
        "Validate classification results with interactive feedback",
        request_model=ValidationRequest
    )
    async def validate_classification(self, request: ValidationRequest):
        """Validate classification results with interactive feedback."""
        # Get classified documents
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.drive_service.files().list(
                q="properties has { key='classified' and value='true' }",
                fields="files(id, name, properties)"
            ).execute()
        )
        
        files = response.get('files', [])
        
        # Select sample based on validation type
        if request.validation_type == "random":
            import random
            sample = random.sample(files, min(request.sample_size, len(files)))
        elif request.validation_type == "category" and request.category:
            sample = [
                f for f in files
                if request.category in f.get('properties', {}).get('categories', '').split(',')
            ][:request.sample_size]
        else:
            sample = files[:request.sample_size]
        
        validation_results = {
            'sample_size': len(sample),
            'total_documents': len(files),
            'results': []
        }
        
        # Process each sample
        for file in sample:
            # Get document content
            content = await self._get_document_content(file['id'])
            
            # Get current classification
            current_classification = {
                'summary': file['properties'].get('classification_summary'),
                'confidence': int(file['properties'].get('overall_confidence', 0)),
                'categories': file['properties'].get('categories', '').split(',')
            }
            
            # Perform new classification
            new_classification = await self._classify_content(
                content,
                {'id': file['id'], 'name': file['name']}
            )
            
            # Compare classifications
            comparison = {
                'document_id': file['id'],
                'name': file['name'],
                'current_classification': current_classification,
                'new_classification': new_classification,
                'differences': self._compare_classifications(
                    current_classification,
                    new_classification
                )
            }
            
            validation_results['results'].append(comparison)
        
        return validation_results

    def _compare_classifications(self, current: Dict, new: Dict) -> Dict:
        """Compare two classifications and identify differences."""
        differences = {
            'confidence_change': new['overall_confidence'] - current['confidence'],
            'category_changes': {
                'added': [c for c in new['categories'] if c not in current['categories']],
                'removed': [c for c in current['categories'] if c not in new['categories']]
            },
            'summary_changed': current['summary'] != new['summary']
        }
        
        differences['significant_changes'] = (
            abs(differences['confidence_change']) > 20 or
            differences['category_changes']['added'] or
            differences['category_changes']['removed']
        )
        
        return differences

    @register_function(
        "Generate custom classification report",
        request_model=CustomReportRequest
    )
    async def generate_custom_report(self, request: CustomReportRequest):
        """Generate a custom classification report with advanced filtering and grouping."""
        # Build query for classified documents
        query_parts = ["properties has { key='classified' and value='true' }"]
        
        if request.filters:
            for key, value in request.filters.items():
                if key == 'confidence_min':
                    query_parts.append(
                        f"properties has {{ key='overall_confidence' and value >= '{value}' }}"
                    )
                elif key == 'confidence_max':
                    query_parts.append(
                        f"properties has {{ key='overall_confidence' and value <= '{value}' }}"
                    )
                elif key == 'categories':
                    categories = value if isinstance(value, list) else [value]
                    category_queries = [
                        f"properties has {{ key='categories' and value contains '{cat}' }}"
                        for cat in categories
                    ]
                    query_parts.append(f"({' or '.join(category_queries)})")
                elif key == 'date_after':
                    query_parts.append(f"properties has {{ key='classification_date' and value >= '{value}' }}")
                elif key == 'date_before':
                    query_parts.append(f"properties has {{ key='classification_date' and value <= '{value}' }}")
        
        query = " and ".join(query_parts)
        
        # Get classified documents
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: self.drive_service.files().list(
                q=query,
                fields="files(id, name, properties)",
                pageSize=1000
            ).execute()
        )
        
        files = response.get('files', [])
        
        # Process and group results
        results = self._process_report_results(files, request.grouping)
        
        # Generate report in requested format
        if request.format == "markdown":
            report = self._generate_markdown_report(
                results,
                request.grouping,
                request.include_details
            )
        else:
            report = json.dumps(results, indent=2)
        
        return {
            'report': report,
            'generated_at': datetime.now().isoformat(),
            'format': request.format,
            'total_documents': len(files)
        }

    def _process_report_results(self, files: List[Dict], grouping: Optional[List[str]]) -> Dict:
        """Process and group report results."""
        results = {
            'summary': {
                'total_documents': len(files),
                'confidence_levels': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'categories': {}
            },
            'groups': {}
        }
        
        for file in files:
            props = file.get('properties', {})
            
            # Update confidence levels
            confidence = int(props.get('overall_confidence', 0))
            if confidence >= self.config.confidence_thresholds['high']:
                results['summary']['confidence_levels']['HIGH'] += 1
            elif confidence >= self.config.confidence_thresholds['medium']:
                results['summary']['confidence_levels']['MEDIUM'] += 1
            else:
                results['summary']['confidence_levels']['LOW'] += 1
            
            # Update categories
            for category in props.get('categories', '').split(','):
                if category:
                    results['summary']['categories'][category] = \
                        results['summary']['categories'].get(category, 0) + 1
            
            # Group results if grouping is specified
            if grouping:
                current_group = results['groups']
                for group_key in grouping:
                    group_value = self._get_group_value(file, group_key)
                    if group_value not in current_group:
                        current_group[group_value] = {
                            'documents': [],
                            'subgroups': {}
                        }
                    current_group = current_group[group_value]['subgroups']
                
                # Add document to its group
                current_group['documents'].append({
                    'id': file['id'],
                    'name': file['name'],
                    'classification': props
                })
        
        return results

    def _get_group_value(self, file: Dict, group_key: str) -> str:
        """Get the value for a grouping key from a file."""
        props = file.get('properties', {})
        
        if group_key == 'confidence_level':
            confidence = int(props.get('overall_confidence', 0))
            if confidence >= self.config.confidence_thresholds['high']:
                return 'HIGH'
            elif confidence >= self.config.confidence_thresholds['medium']:
                return 'MEDIUM'
            return 'LOW'
        elif group_key == 'category':
            categories = props.get('categories', '').split(',')
            return categories[0] if categories else 'Uncategorized'
        elif group_key == 'date':
            date = props.get('classification_date', '').split('T')[0]
            return date or 'Unknown'
        
        return str(file.get(group_key, 'Unknown'))

    def _generate_markdown_report(
        self,
        results: Dict,
        grouping: Optional[List[str]],
        include_details: bool
    ) -> str:
        """Generate a markdown format report."""
        report_parts = [
            "# Document Classification Report",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\nTotal Documents: {results['summary']['total_documents']}",
            "\n## Summary",
            "\n### Confidence Levels",
            *[f"- {level}: {count}" for level, count in results['summary']['confidence_levels'].items()],
            "\n### Categories",
            *[f"- {cat}: {count}" for cat, count in results['summary']['categories'].items()]
        ]
        
        if grouping and results['groups']:
            report_parts.append("\n## Grouped Results")
            self._add_grouped_results(
                report_parts,
                results['groups'],
                grouping,
                include_details,
                level=0
            )
        
        return "\n".join(report_parts)

    def _add_grouped_results(
        self,
        report_parts: List[str],
        groups: Dict,
        grouping: List[str],
        include_details: bool,
        level: int
    ):
        """Add grouped results to the markdown report."""
        indent = "  " * level
        
        for group_name, group_data in groups.items():
            report_parts.append(f"\n{indent}### {grouping[level]}: {group_name}")
            
            if include_details and 'documents' in group_data:
                for doc in group_data['documents']:
                    report_parts.append(
                        f"\n{indent}- {doc['name']} "
                        f"(Confidence: {doc['classification'].get('overall_confidence')}%)"
                    )
            
            if 'subgroups' in group_data and level + 1 < len(grouping):
                self._add_grouped_results(
                    report_parts,
                    group_data['subgroups'],
                    grouping,
                    include_details,
                    level + 1
                )

if __name__ == "__main__":
    server = GDriveClassifierMCP()
    server.start()