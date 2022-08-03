-- Following the given guidelines, we decide to create 5 tables in our new schema:
-- 1) "users" table
-- 2) "topics" table
-- 3) "posts" table
-- 4) "comments" table
-- 5) "votes" table

--------------------------------------------------------------------------
-- Creating table for storing user information
CREATE TABLE "users" (
    "id" SERIAL PRIMARY KEY,
    "username" VARCHAR(25) UNIQUE NOT NULL,
    "last_login" TIMESTAMP,
    CONSTRAINT "check_usrname_len" CHECK (
        LENGTH( TRIM("username")) > 0
    ) 
);
-- Adding the index on "username" column for performant query
CREATE INDEX "find_users_by_usrname" ON "users" ("username");
-- Since unique constraint is case sensiive it allows "Shri1" and "shri1" 
-- as valid usernames. To have case insensitive usernames, we put following
-- contstraint:
CREATE UNIQUE INDEX "check_case_insensitive_username" ON "users"(LOWER("username"));
-------------------------------------------------------------------------

-- Creating table for storing topics information
CREATE TABLE "topics" (
    "id" SERIAL PRIMARY KEY,
    "topic_name" VARCHAR(30) UNIQUE NOT NULL,
    "description" VARCHAR(500),
    -- put constraint for topic_name length > 0
    CONSTRAINT "check_tpname_len" CHECK (LENGTH( TRIM("topic_name")) > 0)    
);
-- create index on topic_name column
CREATE INDEX "find_topic_by_tpname" ON "topics" ("topic_name");
-- in order to create a case insensitve topic_name put constraint
-- this will avoid "topic1" to be different than "Topic1".
CREATE UNIQUE INDEX "check_unique_tpname" ON "topics"(LOWER("topic_name"));

--------------------------------------------------------------------------
-- Create a table posts for storing the post info on existing topics
CREATE TABLE "posts" (
    "id" SERIAL PRIMARY KEY,
    "post_title" VARCHAR(100) NOT NULL,
    "topic_id" INTEGER NOT NULL, -- revised to "not null" as per feedback.
    "user_id" INTEGER, --user-id who posted it
    "url" VARCHAR(4000) DEFAULT NULL,
    "text_content" TEXT DEFAULT NULL, 
    "posting_time" TIMESTAMP,
    -- constraints to be imposed
    -- 1) If a topic gets deleted, all the posts associated with it should be 
    --    automatically deleted too.
    FOREIGN KEY ("topic_id") REFERENCES "topics"("id") ON DELETE CASCADE,
    -- 2) If the user who created the post gets deleted, then the post will 
    --remain, but it will become dissociated from that user. 
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL,
    -- 3) Posts should contain either a URL or a text content, but not both.
    CONSTRAINT "url_or_content" CHECK (
        ("url" IS NOT NULL AND "text_content" IS NULL) OR 
        ("url" IS NULL AND "text_content" IS NOT NULL)
    ),
    -- 4) Title of the post can't be empty
    CONSTRAINT "check_post_title_len" CHECK (
        LENGTH( TRIM("post_title")) > 0
    )
);
-- Creating index for this table
CREATE INDEX "search_posts_by_user" ON "posts"("user_id");
CREATE INDEX "search_posts_by_topic" ON "posts"("topic_id");
CREATE INDEX "url_specific_posts" ON "posts"("url");
----------------------------------------------------------------------------------
-- creating the comments table
CREATE TABLE "comments" (
    "id" SERIAL PRIMARY KEY,
    "user_id" INTEGER,
    "post_id" INTEGER NOT NULL,
    "parent_comment_id" INTEGER DEFAULT NULL, 
    "comment_text" TEXT NOT NULL,
    "comment_post_time" TIMESTAMP,

    -- putting constraints
    -- 1) comment should belong to the valid user 
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL,
    -- 2) comment should belong to the valid post id 
    FOREIGN KEY ("post_id") REFERENCES "posts"("id") ON DELETE CASCADE,
    -- 3) comment should have a valid parent comment id 
    FOREIGN KEY ("parent_comment_id") REFERENCES "comments"("id") ON DELETE CASCADE,
    -- 4) constraint for "a comments text cant be empty"
    CONSTRAINT "comment_text_not_empty" CHECK (
        LENGTH( TRIM("comment_text")) > 0
    )    
);

CREATE INDEX "find_comments_by_user" ON "comments"("user_id");
CREATE INDEX "find_comments_by_post" ON "comments"("post_id");
CREATE INDEX "find_comments_by_parent_id" ON "comments"("parent_comment_id");
------------------------------------------------------------------------------------------
-- creating the votes table
CREATE TABLE "votes" (
    "id" SERIAL PRIMARY KEY,
    "user_id" INTEGER,
    "post_id" INTEGER NOT NULL,
    "vote_value" SMALLINT NOT NULL,

    -- putting constraints
    -- 1) vote should be given by the valid user 
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL,
    -- 2) vote should be considered for the valid post id 
    FOREIGN KEY ("post_id") REFERENCES "posts"("id") ON DELETE CASCADE,
    -- 3) constraint for "value of vote must be either 1 or -1"
    CONSTRAINT "check_valid_vote_value" CHECK (
        "vote_value" = 1 OR "vote_value" = -1
        )
    );
CREATE INDEX "find_votes_by_post" ON "votes"("post_id");

-----------------------------------------------------------------------------------------



