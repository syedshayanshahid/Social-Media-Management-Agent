import os
import datetime
import time
import traceback
from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client, Client
from arcadepy import Arcade, AsyncArcade
import streamlit as st
import asyncio
from agents import Agent, Runner
from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError
import threading
import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
# Load only the real .env file and suppress parse warnings
try:
    load_dotenv('.env', override=True, verbose=False)
except Exception:
    logging.warning("Could not load .env file or parse warnings occurred.")

# Fetch environment variables with safe defaults
USER_ID_FOR_LINKEDIN = os.getenv("LINKEDIN_USER_ID", "shayansynshahid@gmail.com")

# Define LinkedIn tool name constant only
LINKEDIN_TOOL_NAME = "LinkedIn.CreateTextPost"

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

class SocialMediaAgent(Agent):
    def __init__(self, openai_client: OpenAI, arcade_linkedin: Arcade, supabase: Client):
        """Initialize the SocialMediaAgent with OpenAI, Arcade, and Supabase clients."""
        super().__init__(name="SocialMediaAgent")
        self.openai_client = openai_client
        self.arcade_linkedin = arcade_linkedin
        self.supabase = supabase

    def web_search_preview(self, prompt: str):
        """Run the agent synchronously via Runner to get search results."""
        runner = Runner()
        return asyncio.run(runner.run(self, prompt))

    def generate_post(self, content: str) -> str:
        """Generate a LinkedIn post based on the provided content."""
        prompt = (
            "Write a professional and engaging LinkedIn post based on the provided content:\n"
            f"{content}\n"
            "Keep it concise and use a friendly tone appropriate for LinkedIn."
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes professional LinkedIn posts."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def save_scheduled_post(self, post_text: str, platforms: list, scheduled_at: datetime.datetime) -> None:
        """Save a scheduled post to Supabase."""
        try:
            self.supabase.table("scheduled_posts").insert({
                "post_text": post_text,
                "platforms": platforms,
                "scheduled_at": scheduled_at.isoformat(),
                "is_posted": False,
                "exec_response": {
                    "status": "pending",
                    "scheduled_for": scheduled_at.isoformat()
                }
            }).execute()
        except SQLAlchemyError as e:
            # Handle SQLAlchemy errors (e.g., table not found)
            if '42P01' in str(e):
                st.error("Missing scheduled_posts table. Create it in Supabase first!")
            else:
                # Log other SQL errors
                logging.error(f"Database error saving scheduled post: {e}")

    def get_due_posts(self) -> list:
        """Retrieve posts due for publishing with error handling"""
        try:
            result = self.supabase.table("scheduled_posts").select("*")\
                .lt("scheduled_at", datetime.datetime.now().isoformat())\
                .eq("is_posted", False)\
                .execute()
            return result.data
        except SQLAlchemyError as e:
            # Handle SQLAlchemy errors (e.g., table not found)
            if '42P01' in str(e):
                st.error("Missing scheduled_posts table. Create it in Supabase first!")
                return []
            # Other API errors
            print(f"Database error fetching due posts: {e}")
            return []

# AsyncArcade OpenAI agent setup (you can keep this if you still need it)
async def init_openai_agent() -> Agent:
    client = AsyncArcade()
    tools = await get_arcade_tools(client, ["openai"])
    return Agent(
        name="OpenAI agent",
        instructions="You are a helpful assistant that can assist with OpenAI API calls.",
        model="gpt-4o-mini",
        tools=tools,
    )

# Runner for the OpenAI agent
async def run_openai_agent_example(user_input: str):
    try:
        openai_agent = await init_openai_agent()
        runner = Runner()
        result = await runner.run(openai_agent, user_input)
        print(result)
    except AuthorizationError as e:
        print(f"Please visit this URL to authorize: {e}")

def background_worker():
    """Run in a pure Python thread without Streamlit dependencies"""
    backoff = 1
    while True:
        try:
            # Reload env vars
            load_dotenv()
            sup_url = os.getenv("SUPABASE_URL")
            sup_key = os.getenv("SUPABASE_KEY")
            if not sup_url or not sup_key:
                raise RuntimeError("Missing Supabase credentials")
            supabase_client = create_client(sup_url, sup_key)

            arcade_linkedin = Arcade()

            # Raw fetch of due posts with table existence check
            try:
                posts = supabase_client.table("scheduled_posts").select("*") \
                    .lt("scheduled_at", datetime.datetime.now().isoformat()) \
                    .eq("is_posted", False)\
                    .execute().data or []
            except SQLAlchemyError as e:
                # Handle SQLAlchemy errors (e.g., table not found)
                if '42P01' in str(e):
                    print("Worker warning: 'scheduled_posts' table does not exist yet. Skipping iteration.")
                    time.sleep(60)
                    continue
                else:
                    raise
            # Process each post
            for post in posts:
                post_success = False
                # Skip already posted
                if post.get('is_posted', False):
                    print(f"Post ID {post['id']} already marked as posted. Skipping.")
                    continue

                # LinkedIn posting
                if "LinkedIn" in post.get('platforms', []):
                    try:
                        # First update exec_response to indicate we're attempting LinkedIn posting
                        now_iso = datetime.datetime.now().isoformat()
                        existing_exec_response = post.get('exec_response', {})
                        if not isinstance(existing_exec_response, dict):
                            existing_exec_response = {}
                            
                        # Update status to attempting
                        linkedin_attempt_exec_response = {
                            **existing_exec_response,
                            "linkedin_posting_attempt": now_iso,
                            "execution_status": "in_progress"
                        }
                        
                        try:
                            supabase_client.table('scheduled_posts').update({
                                'exec_response': linkedin_attempt_exec_response,
                                'last_updated': now_iso
                            }).eq('id', post['id']).execute()
                        except Exception as update_err:
                            print(f"Error updating exec_response before LinkedIn posting: {update_err}")
                        
                        # Authorize with retry logic for potential network issues
                        max_auth_retries = 2
                        for auth_attempt in range(max_auth_retries):
                            try:
                                auth = arcade_linkedin.tools.authorize(
                                    tool_name=LINKEDIN_TOOL_NAME,
                                    user_id=USER_ID_FOR_LINKEDIN
                                )
                                if auth.status != "completed":
                                    print(f"LinkedIn authorization required for post ID: {post['id']}")
                                    print(f"Auth status: {auth.status}, URL: {getattr(auth, 'url', 'No URL')}")
                                    
                                    # Update exec_response with auth needed info
                                    auth_needed_resp = {
                                        **linkedin_attempt_exec_response,
                                        "execution_status": "auth_needed",
                                        "linkedin_auth_url": getattr(auth, 'url', None),
                                        "last_auth_check": now_iso
                                    }
                                    
                                    try:
                                        supabase_client.table('scheduled_posts').update({
                                            'exec_response': auth_needed_resp,
                                            'last_updated': now_iso
                                        }).eq('id', post['id']).execute()
                                    except Exception as auth_update_err:
                                        print(f"Error updating LinkedIn auth needed status: {auth_update_err}")
                                        
                                    if auth_attempt < max_auth_retries - 1:
                                        print(f"Retrying LinkedIn authorization in 5 seconds...")
                                        time.sleep(5)
                                        continue
                                    break
                                break
                            except Exception as auth_e:
                                print(f"LinkedIn authorization error: {auth_e}")
                                if auth_attempt < max_auth_retries - 1:
                                    print(f"Retrying LinkedIn authorization in 5 seconds...")
                                    time.sleep(5)
                                    continue
                                raise
                        
                        if auth.status != "completed":
                            print(f"LinkedIn authorization failed after {max_auth_retries} attempts for post ID: {post['id']}")
                            continue
                            
                        arcade_linkedin.auth.wait_for_completion(auth)
                        
                        # Execute with retry for potential rate limits
                        max_exec_retries = 2
                        for exec_attempt in range(max_exec_retries):
                            try:
                                exec_resp = arcade_linkedin.tools.execute(
                                    tool_name=LINKEDIN_TOOL_NAME,
                                    input={'text': post['post_text']},
                                    user_id=USER_ID_FOR_LINKEDIN
                                )
                                break
                            except Exception as exec_e:
                                if "429" in str(exec_e) and exec_attempt < max_exec_retries - 1:
                                    print(f"LinkedIn rate limit hit, retrying in 15 seconds...")
                                    time.sleep(15)
                                    continue
                                raise
                        
                        # Check for successful post
                        if getattr(exec_resp, "error", None):
                            err_obj = exec_resp.error
                            err_msg = str(err_obj)
                            print(f"LinkedIn posting error for post ID {post['id']}: {err_msg}")
                            
                            # Update exec_response with error details
                            error_resp = {
                                **linkedin_attempt_exec_response,
                                "execution_status": "error",
                                "linkedin_error": {
                                    "message": err_msg,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                            }
                            
                            try:
                                supabase_client.table('scheduled_posts').update({
                                    'exec_response': error_resp,
                                    'last_updated': datetime.datetime.now().isoformat(),
                                    'last_error': str({"platform": "LinkedIn", "message": err_msg})
                                }).eq('id', post['id']).execute()
                            except Exception as err_update_e:
                                print(f"Error updating LinkedIn error status: {err_update_e}")
                        else:
                            post_url = getattr(exec_resp, "url", None)
                            if post_url:
                                print(f"LinkedIn post successful! URL: {post_url}")
                                post_success = True
                                
                                # Update exec_response with success details
                                success_resp = {
                                    **linkedin_attempt_exec_response,
                                    "execution_status": "success",
                                    "linkedin_success": {
                                        "url": post_url,
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                }
                                
                                try:
                                    supabase_client.table('scheduled_posts').update({
                                        'exec_response': success_resp,
                                        'last_updated': datetime.datetime.now().isoformat()
                                    }).eq('id', post['id']).execute()
                                except Exception as success_update_e:
                                    print(f"Error updating LinkedIn success status: {success_update_e}")
                            else:
                                print(f"LinkedIn post executed but no URL returned for post ID: {post['id']}")
                                post_success = True
                                
                                # Update with generic success
                                success_resp = {
                                    **linkedin_attempt_exec_response,
                                    "execution_status": "success",
                                    "linkedin_success": {
                                        "timestamp": datetime.datetime.now().isoformat()
                                    }
                                }
                                
                                try:
                                    supabase_client.table('scheduled_posts').update({
                                        'exec_response': success_resp,
                                        'last_updated': datetime.datetime.now().isoformat()
                                    }).eq('id', post['id']).execute()
                                except Exception as success_update_e:
                                    print(f"Error updating LinkedIn success status: {success_update_e}")
                    except Exception as e:
                        print(f"Worker error posting LinkedIn: {e}")
                        error_msg = str(e)
                        
                        # Update exec_response with error details
                        existing_exec_response = post.get('exec_response', {})
                        if not isinstance(existing_exec_response, dict):
                            existing_exec_response = {}
                            
                        error_resp = {
                            **existing_exec_response,
                            "execution_status": "error",
                            "linkedin_error": {
                                "message": error_msg,
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        }
                        
                        # Special handling for permission errors (403)
                        if "403" in error_msg:
                            error_resp["execution_status"] = "permission_error"
                            error_resp["linkedin_error"]["details"] = "403 Forbidden - Missing permissions"
                            print(f"LinkedIn permission error (403) for post ID: {post['id']}. Needs w_member_social scope.")
                            
                        try:
                            supabase_client.table('scheduled_posts').update({
                                'exec_response': error_resp,
                                'last_updated': datetime.datetime.now().isoformat(),
                                'last_error': str({"platform": "LinkedIn", "message": error_msg})
                            }).eq('id', post['id']).execute()
                        except Exception as err_update_e:
                            print(f"Error updating LinkedIn error status: {err_update_e}")
                # Mark as posted if LinkedIn succeeded
                if post_success:
                    try:
                        # Add result info to the database
                        now = datetime.datetime.now()
                        now_iso = now.isoformat()
                        
                        # Get existing exec_response or create a new one
                        existing_exec_response = post.get('exec_response', {})
                        if not isinstance(existing_exec_response, dict):
                            existing_exec_response = {}
                            
                        # Update the execution response with success details
                        updated_exec_response = {
                            **existing_exec_response,
                            "status": "completed",
                            "execution_status": "success",
                            "posted_at": now_iso,
                            "execution_details": {
                                "platforms_posted": post.get('platforms', []),
                                "success": True,
                                "timestamp": now_iso
                            }
                        }
                        
                        # Add URL if available
                        if hasattr(exec_resp, 'url') and exec_resp.url:
                            updated_exec_response["post_url"] = exec_resp.url
                            
                        # Update with more detailed information
                        supabase_client.table('scheduled_posts').update({
                            'is_posted': True,
                            'posted_at': now_iso,
                            'post_result': 'success',
                            'last_updated': now_iso,
                            'exec_response': updated_exec_response
                        }).eq('id', post['id']).execute()
                        
                        print(f"Post ID {post['id']} marked as posted successfully at {now_iso}")
                        print(f"Execution response updated with success status")
                    except Exception as e:
                        print(f"Worker error marking posted: {e}")
                        print(f"Failed to update database for post ID {post['id']}")
            
            # Reset backoff on successful iteration
            backoff = 1
        except Exception as e:
            error_msg = str(e)
            print(f"Worker critical error: {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            # Exponential backoff with cap
            time.sleep(min(300, backoff))
            backoff = min(backoff * 2, 300)
        finally:
            # Always wait between iterations
            time.sleep(60)

def select_scheduled_datetime():
    """
    Presents a Streamlit UI to select a scheduled date and time.
    Returns:
    scheduled_datetime (datetime.datetime or None): The chosen datetime if valid,
    otherwise None.
    """
    # Date picker
    scheduled_date = st.date_input(
        "Select scheduled date",
        datetime.date.today()
    )
    st.write("Select scheduled time:")
    # Hour, minute, and AM/PM selectors
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        hour_choices = list(range(1, 13))
        current_hour = datetime.datetime.now().hour
        default_hour = current_hour % 12 or 12
        hour = st.selectbox("Hour", hour_choices, index=hour_choices.index(default_hour))
    with col2:
        minute_choices = [f"{i:02d}" for i in range(0, 60)]
        default_minute = f"{datetime.datetime.now().minute:02d}"
        minute = st.selectbox("Minute", minute_choices, index=minute_choices.index(default_minute))
    with col3:
        period = st.selectbox("AM/PM", ["AM", "PM"], index=0 if current_hour < 12 else 1)

    # Convert to 24-hour time
    hour_24 = (hour % 12) + (12 if period == "PM" else 0)
    scheduled_time = datetime.time(hour=hour_24, minute=int(minute))
    scheduled_datetime = datetime.datetime.combine(scheduled_date, scheduled_time)

    if scheduled_datetime <= datetime.datetime.now():
        st.error("Scheduled time must be in the future.")
        return None

    st.write(f"Will post at: {scheduled_datetime.strftime('%Y-%m-%d %I:%M %p')}")
    return scheduled_datetime

def render_post_input_fields(key_prefix: str):
    content_type = st.selectbox(
        "Content Type",
        ["Original post", "Article summary", "Announcement"],
        index=0,
        key=f"{key_prefix}_content_type"
    )
    topic = st.text_input(
        "Please enter a prompt to generate content", "",
        key=f"{key_prefix}_topic"
    )
    tone = st.selectbox(
        "Tone",
        ["Professional", "Casual", "Enthusiastic"],
        index=0,
        key=f"{key_prefix}_tone"
    )
    platforms = st.multiselect(
        "Select platform to post", ["LinkedIn"],
        key=f"{key_prefix}_platforms"
    )
    return content_type, topic, tone, platforms

def main() -> None:
    st.title("Social Media Agent Content Scheduler 🤖")
    
    # Create two tabs for immediate and scheduled posting
    tab_immediate, tab_scheduled = st.tabs(["Post Now", "Schedule Post"])
    
    # Initialize OpenAI and Arcade clients outside the tabs
    _openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    arcade_linkedin = Arcade()
    supabase_client = create_client(url, key)
    agent = SocialMediaAgent(
        _openai_client,
        arcade_linkedin,
        supabase_client
    )
    
    # IMMEDIATE POSTING TAB
    with tab_immediate:
        st.header("Post Immediately")
        st.write("Create and post content immediately to your social media platforms.")
        
        content_type, topic, tone, platforms = render_post_input_fields("immediate")
        
        if st.button("Post Now"):
            # Validation
            if not topic:
                st.error("Please enter a topic or key points to generate content.")
                return
            if not platforms:
                st.error("Please select at least one platform to post.")
                return
                
            # Generate content
            with st.spinner("Generating post content..."):
                full_prompt = f"{content_type}: {topic} (Tone: {tone})"
                post_text = agent.generate_post(full_prompt)
            
            # Use current time for immediate posting
            post_time = datetime.datetime.now()
            
            # Save to Supabase (this will be picked up by the worker thread)
            agent.save_scheduled_post(post_text, platforms, post_time)
            formatted_time = post_time.strftime('%Y-%m-%d %I:%M %p')
            st.success(f"Post queued for immediate posting to {', '.join(platforms)}! 🚀")
            
            # Show preview
            with st.expander("Post Preview", expanded=True):
                st.subheader("Post Content")
                st.write(post_text)
                st.caption(f"Queued for posting at: {formatted_time}")
            
            # LinkedIn
            if "LinkedIn" in platforms:
                st.subheader("LinkedIn Posting")
                st.info("Posting to LinkedIn…")

                # Step 1: Authorize
                auth = arcade_linkedin.tools.authorize(
                    tool_name=LINKEDIN_TOOL_NAME,
                    user_id=USER_ID_FOR_LINKEDIN,
                )
                if auth.status != "completed":
                    st.warning("Authorization required. Please authorize and then retry.")
                    st.link_button("Authorize LinkedIn", auth.url)
                    return
                arcade_linkedin.auth.wait_for_completion(auth)

                # Step 2: Execute post
                try:
                    exec_resp = arcade_linkedin.tools.execute(
                        tool_name=LINKEDIN_TOOL_NAME,
                        input={"text": post_text},
                        user_id=USER_ID_FOR_LINKEDIN,
                    )
                    post_url = getattr(exec_resp, "url", None)
                    if post_url:
                        st.success("LinkedIn post successful!")
                        st.link_button("View LinkedIn post", post_url)
                    else:
                        st.success("LinkedIn post sent.")
                        st.write(exec_resp)
                except Exception as e:
                    st.error(f"Error posting LinkedIn: {e}")
                    st.info("Please re-authorize or check your API access.")
    
    # SCHEDULED POSTING TAB
    with tab_scheduled:
        st.header("Schedule Post for Later")
        st.write("Create content now and schedule it to be posted at a specific time.")
        
        content_type, topic, tone, platforms = render_post_input_fields("scheduled")
        
        # Get scheduled time using the helper function
        scheduled_datetime = select_scheduled_datetime()
        
        if st.button("Schedule Post"):
            # Validation
            if not topic:
                st.error("Please enter a topic or key points to generate content.")
                return
            if not platforms:
                st.error("Please select at least one platform to post.")
                return
            if scheduled_datetime is None or scheduled_datetime <= datetime.datetime.now():
                st.error("Please select a valid future date and time.")
                return
                
            # Generate content
            with st.spinner("Generating post content..."):
                full_prompt = f"{content_type}: {topic} (Tone: {tone})"
                post_text = agent.generate_post(full_prompt)
            
            # Save scheduled post to Supabase
            agent.save_scheduled_post(post_text, platforms, scheduled_datetime)
            formatted_time = scheduled_datetime.strftime('%Y-%m-%d %I:%M %p')
            st.success(f"Post scheduled for {formatted_time} to {', '.join(platforms)}! 🗓️")
            
            # Show preview
            with st.expander("Scheduled Post Preview", expanded=True):
                st.subheader("Post Content")
                st.write(post_text)
                st.caption(f"Scheduled for: {formatted_time}")

            # Fetch and display execution result once the background worker has run
            # Instead, we’ll try to post to LinkedIn in‐place at the scheduled time
            if "LinkedIn" in platforms:
                # calculate how long to wait until the scheduled time
                delay = (scheduled_datetime - datetime.datetime.now()).total_seconds()
                if delay > 0:
                    st.info(f"Waiting {int(delay)} seconds until {formatted_time} to post on LinkedIn…")
                    time.sleep(delay)

                try:
                    exec_resp = arcade_linkedin.tools.execute(
                        tool_name=LINKEDIN_TOOL_NAME,
                        input={"text": post_text},
                        user_id=USER_ID_FOR_LINKEDIN,
                    )
                    post_url = getattr(exec_resp, "url", None)
                    if post_url:
                        st.success("LinkedIn post successful!")
                        st.link_button("View LinkedIn post", post_url)
                    else:
                        st.success("LinkedIn post sent.")
                        st.write(exec_resp)
                except Exception as e:
                    st.error(f"Error posting LinkedIn: {e}")
                    st.info("Please re-authorize or check your API access.")

if __name__ == "__main__":
    # Launch Streamlit app and worker once
    if not hasattr(st, '_worker_started'):
        worker = threading.Thread(
            target=background_worker,
            daemon=True,
            name="post_scheduler_worker"
        )
        worker.start()
        st._worker_started = True
    main()
