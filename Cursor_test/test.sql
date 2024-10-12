-- 获取总数、均值和中位数
SELECT
    COUNT(*) AS totalnum,
    ROUND(AVG(index_value), 6) AS valuemean,
    ROUND(SUBSTRING_INDEX(SUBSTRING_INDEX(GROUP_CONCAT(index_value ORDER BY index_value), ',', FLOOR(COUNT(*) / 2) + 1), ',', -1), 6) AS valuemedian
INTO @totalnum, @valuemean, @valuemedian
FROM (
  SELECT str_id, index_value
  FROM ADS_PF_FUND_INDICATOR_HIS_1
  WHERE second_type_code = '100103'
    AND enddate = '20100131'
    AND period_code = 'M'
    AND index_code = 'GeoRelare'
  UNION ALL
  SELECT str_id, index_value
  FROM ADS_PF_FUND_INDICATOR_HIS_2
  WHERE second_type_code = '100103'
    AND enddate = '20100131'
    AND period_code = 'M'
    AND index_code = 'GeoRelare'
) AS combined_data;

-- 获取排名
SELECT
    str_id,
    index_value,
    @rank := @rank + 1 AS ranking,
    @totalnum AS totalnum,
    @valuemean AS valuemean,
    @valuemedian AS valuemedian
FROM (
  SELECT str_id, index_value
  FROM (
    SELECT str_id, index_value
    FROM ADS_PF_FUND_INDICATOR_HIS_1
    WHERE second_type_code = '100103'
      AND enddate = '20100131'
      AND period_code = 'M'
      AND index_code = 'GeoRelare'
    UNION ALL
    SELECT str_id, index_value
    FROM ADS_PF_FUND_INDICATOR_HIS_2
    WHERE second_type_code = '100103'
      AND enddate = '20100131'
      AND period_code = 'M'
      AND index_code = 'GeoRelare'
  ) AS combined_data
  ORDER BY index_value DESC
) AS ranked_data, (SELECT @rank := 0) AS r;


