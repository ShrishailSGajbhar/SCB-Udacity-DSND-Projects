-- Data Migration from bad_schema to our schema
-------------------------------------------------
-- 1) Migrating the users considering the following guideline
--   "Don’t forget that some users only vote or comment, and haven’t created any posts. 
--    You’ll have to create those users too."

INSERT INTO "users" ("username") (
    SELECT DISTINCT "username" FROM "bad_posts"
    UNION 
    SELECT DISTINCT "username" FROM "bad_comments"
    UNION 
    SELECT DISTINCT REGEXP_SPLIT_TO_TABLE("upvotes", ',') FROM "bad_posts"
    UNION 
    SELECT DISTINCT REGEXP_SPLIT_TO_TABLE("downvotes", ',') FROM "bad_posts"
);

-------------------------------------------------
-- 2) Migrating the topics from bad_schema
INSERT INTO "topics" ("topic_name") (
    SELECT DISTINCT "topic" FROM "bad_posts"
);

-------------------------------------------------
-- Migrating the posts from the bad_schema
INSERT INTO "posts" ("post_title","topic_id","user_id","url","text_content") (
    (
        SELECT SUBSTRING("b"."title",1,100), "t"."id", "u"."id",
        "b"."url","b"."text_content"
        FROM "bad_posts" "b"
        JOIN "topics" "t"
        ON "b"."topic"="t"."topic_name"
        JOIN "users" "u"
        ON "b"."username"="u"."username"
    ) 
);
--------------------------------------------------
-- Migrating the comments from the bad_schema
INSERT INTO "comments" ("user_id","post_id","comment_text") (
    (
        SELECT  "u"."id", "p"."id", "b"."text_content"
        FROM "bad_comments" "b"
        JOIN "users" "u"
        ON "b"."username"="u"."username"
        JOIN "posts" "p"
        ON "b"."post_id"="p"."id"  
    )
);
-----------------------------------------------------
--Migrating the votes (upvotes)
INSERT INTO "votes" ("user_id","post_id","vote_value") (
    SELECT "u"."id", "b"."id", 1 AS "up_vote"
    FROM (
        SELECT "id", REGEXP_SPLIT_TO_TABLE("upvotes",',') AS "upvotes"
        FROM "bad_posts" 
         ) "b"  
    JOIN "users" "u"
    ON "b"."upvotes"="u"."username"
);
--Migrating the votes (downvotes)
INSERT INTO "votes" ("user_id","post_id","vote_value") (
    SELECT "u"."id", "b"."id", -1 AS "down_vote"
    FROM (
        SELECT "id", REGEXP_SPLIT_TO_TABLE("downvotes",',') AS "downvotes"
        FROM "bad_posts" 
         ) "b"  
    JOIN "users" "u"
    ON "b"."downvotes"="u"."username"
);
-------------------------------------------------------
